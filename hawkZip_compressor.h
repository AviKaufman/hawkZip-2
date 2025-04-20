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

            // 1) quantize into a small local buffer
            int qBuf[BLOCK_SIZE];
            for (int i = 0; i < len; i++) {
                float r = oriData[bs + i] * (0.5f / errorBound);
                int sign = (r < -0.5f);
                int q = (int)(r + 0.5f) - sign;
                qBuf[i] = q;
            }

            // 2) delta‑encode Q into absQuant[], build new signFlag
            unsigned int sflag = 0;
            int max_q = 0;

            // element 0
            int delta0 = qBuf[0];
            int a0 = delta0 < 0 ? -delta0 : delta0;
            absQuant[bs] = a0;
            sflag |= (unsigned int)(delta0 < 0)
                     << (BLOCK_SIZE - 1 - (bs % BLOCK_SIZE));
            if (a0 > max_q) max_q = a0;
            int prev = qBuf[0];

            // the rest
            for (int i = 1; i < len; i++) {
                int d = qBuf[i] - prev;
                int ad = d < 0 ? -d : d;
                absQuant[bs + i] = ad;
                sflag |= (unsigned int)(d < 0)
                         << (BLOCK_SIZE - 1 - ((bs + i) % BLOCK_SIZE));
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

