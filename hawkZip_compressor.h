#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

// fixed at 32
#ifndef THREAD_COUNT
#define THREAD_COUNT 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// --- compress kernel with block‑local delta encoding ---
void hawkZip_compress_kernel(
    float* oriData,
    unsigned char* cmpData,
    int* absQuant,
    unsigned int* signFlag,
    int* fixedRate,
    unsigned int* threadOfs,
    size_t nbEle,
    size_t* cmpSize,
    float errorBound)
{
    // Make sure BLOCK_SIZE is a power of 2 for bitwise optimization
    #if ((BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0)
    #error "BLOCK_SIZE must be a power of 2 for bitwise optimization"
    #endif
    
    // Define bit mask for modulo operations
    const int BLOCK_MASK = BLOCK_SIZE - 1;
    
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size  = (nbEle + THREAD_COUNT - 1) / THREAD_COUNT;
    omp_set_num_threads(THREAD_COUNT);

    #pragma omp parallel
    {
        int tid        = omp_get_thread_num();
        int startElem  = tid * chunk_size;
        int endElem    = startElem + chunk_size;
        if (endElem > (int)nbEle) endElem = nbEle;

        int perThread  = (totalBlocks + THREAD_COUNT - 1) / THREAD_COUNT;
        int startBlock = tid * perThread;
        unsigned int local_ofs = 0;

        // each block
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;

            int bs = g * BLOCK_SIZE;
            int be = bs + BLOCK_SIZE;
            if (be > (int)nbEle) be = nbEle;
            int len = be - bs;

            // 1) quantize into a small local buffer - UNROLLED BY 4
            int qBuf[BLOCK_SIZE];
            float inv_err_bound = 0.5f / errorBound;
            
            int i = 0;
            // Process blocks of 4 elements at a time
            for (; i < len - 3; i += 4) {
                // Element 1
                float r1 = oriData[bs + i] * inv_err_bound;
                int sign1 = (r1 < -0.5f);
                qBuf[i] = (int)(r1 + 0.5f) - sign1;
                
                // Element 2
                float r2 = oriData[bs + i + 1] * inv_err_bound;
                int sign2 = (r2 < -0.5f);
                qBuf[i + 1] = (int)(r2 + 0.5f) - sign2;
                
                // Element 3
                float r3 = oriData[bs + i + 2] * inv_err_bound;
                int sign3 = (r3 < -0.5f);
                qBuf[i + 2] = (int)(r3 + 0.5f) - sign3;
                
                // Element 4
                float r4 = oriData[bs + i + 3] * inv_err_bound;
                int sign4 = (r4 < -0.5f);
                qBuf[i + 3] = (int)(r4 + 0.5f) - sign4;
            }
            
            // Handle remaining elements
            for (; i < len; i++) {
                float r = oriData[bs + i] * inv_err_bound;
                int sign = (r < -0.5f);
                qBuf[i] = (int)(r + 0.5f) - sign;
            }

            // 2) delta‑encode Q into absQuant[], build new signFlag
            unsigned int sflag = 0;
            int max_q = 0;

            // element 0
            int delta0 = qBuf[0];
            int a0 = delta0 < 0 ? -delta0 : delta0;
            absQuant[bs] = a0;
            
            // Replace modulo with bitwise AND
            int bit_position = BLOCK_SIZE - 1 - (bs & BLOCK_MASK);
            sflag |= (unsigned int)(delta0 < 0) << bit_position;
            
            if (a0 > max_q) max_q = a0;
            int prev = qBuf[0];

            // the rest
            for (int i = 1; i < len; i++) {
                int d = qBuf[i] - prev;
                int ad = d < 0 ? -d : d;
                absQuant[bs + i] = ad;
                
                // Replace modulo with bitwise AND
                bit_position = BLOCK_SIZE - 1 - ((bs + i) & BLOCK_MASK);
                sflag |= (unsigned int)(d < 0) << bit_position;
                
                if (ad > max_q) max_q = ad;
                prev = qBuf[i];
            }

            signFlag[g] = sflag;

            // 3) determine bit‑width
            int rate = max_q
                       ? ((int)(sizeof(int)*8) - __builtin_clz(max_q))
                       : 0;
            fixedRate[g] = rate;
            cmpData[g]   = (unsigned char)rate;

            if (rate)
                local_ofs += (BLOCK_SIZE + rate * BLOCK_SIZE) / 8;
        }

        threadOfs[tid] = local_ofs;
        #pragma omp barrier

        // prefix-sum to compute write offset
        unsigned int offs = 0;
        for (int t = 0; t < tid; t++)
            offs += threadOfs[t];
        unsigned int write_ptr = offs + totalBlocks;

        // bit‑plane pack the deltas
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;
            int rate = fixedRate[g];
            if (!rate) continue;

            unsigned int sflag = signFlag[g];
            // write 4‑byte sign flag
            for (int byte = 3; byte >= 0; byte--)
                cmpData[write_ptr++] =
                  (sflag >> (8 * byte)) & 0xFF;

            // write bitplanes of absQuant[bs..be)
            int bs = g * BLOCK_SIZE;
            int be = bs + BLOCK_SIZE;
            if (be > (int)nbEle) be = nbEle;
            unsigned int mask = 1;
            for (int bit = 0; bit < rate; bit++) {
                for (int chunk = 0; chunk < BLOCK_SIZE; chunk += 8) {
                    unsigned char packed = 0;
                    int limit = ((bs + chunk + 8) > be)
                                ? (be - (bs + chunk)) : 8;
                    for (int k = 0; k < limit; k++) {
                        packed |=
                          ((absQuant[bs + chunk + k] & mask) >> bit)
                          << (7 - k);
                    }
                    cmpData[write_ptr++] = packed;
                }
                mask <<= 1;
            }
        }

        // last thread writes total compressed size
        if (tid == THREAD_COUNT - 1) {
            unsigned int sum = 0;
            for (int t = 0; t < THREAD_COUNT; t++)
                sum += threadOfs[t];
            *cmpSize = sum + totalBlocks;
        }
    }
}


// --- decompress kernel with delta‑decode ---
void hawkZip_decompress_kernel(
    float* decData,
    unsigned char* cmpData,
    int* absQuant,
    int* fixedRate,
    unsigned int* threadOfs,
    size_t nbEle,
    float errorBound)
{
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size  = (nbEle + THREAD_COUNT - 1) / THREAD_COUNT;
    omp_set_num_threads(THREAD_COUNT);

    #pragma omp parallel
    {
        int tid        = omp_get_thread_num();
        int startElem  = tid * chunk_size;
        int endElem    = startElem + chunk_size;
        if (endElem > (int)nbEle) endElem = nbEle;

        int perThread  = (totalBlocks + THREAD_COUNT - 1) / THREAD_COUNT;
        int startBlock = tid * perThread;
        unsigned int local_ofs = 0;

        // 1) read per‑block rates
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;
            int rate = (int)cmpData[g];
            fixedRate[g] = rate;
            if (rate)
                local_ofs += (BLOCK_SIZE + rate * BLOCK_SIZE) / 8;
        }
        threadOfs[tid] = local_ofs;
        #pragma omp barrier

        // 2) prefix‑sum to get read offset
        unsigned int offs = 0;
        for (int t = 0; t < tid; t++)
            offs += threadOfs[t];
        unsigned int read_ptr = offs + totalBlocks;

        // 3) unpack ‑ decode deltas into absQuant
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;
            int rate = fixedRate[g];
            if (!rate) continue;

            // read sign flag
            unsigned int sflag = 0;
            for (int byte = 3; byte >= 0; byte--)
                sflag |= ((unsigned int)cmpData[read_ptr++])
                         << (8 * byte);

            int bs = g * BLOCK_SIZE;
            int be = bs + BLOCK_SIZE;
            if (be > (int)nbEle) be = nbEle;
            unsigned int mask = 1;

            // clear absQuant for this block
            for (int i = bs; i < be; i++) absQuant[i] = 0;

            // read bitplanes
            for (int bit = 0; bit < rate; bit++) {
                for (int chunk = 0; chunk < BLOCK_SIZE; chunk += 8) {
                    unsigned char packed = cmpData[read_ptr++];
                    int limit = ((bs + chunk + 8) > be)
                                ? (be - (bs + chunk)) : 8;
                    for (int k = 0; k < limit; k++) {
                        absQuant[bs + chunk + k] |=
                          ((packed >> (7 - k)) & 1) << bit;
                    }
                }
                mask <<= 1;
            }

            // 4) delta‑decode + dequantize into decData
            int prevQ = 0;
            for (int i = bs; i < be; i++) {
                int sign = (sflag >> (BLOCK_SIZE - 1 - (i % BLOCK_SIZE))) & 1;
                int d = sign ? -absQuant[i] : absQuant[i];
                if (i == bs) prevQ = d;
                else          prevQ += d;
                decData[i] = prevQ * errorBound * 2;
            }
        }
    }
}