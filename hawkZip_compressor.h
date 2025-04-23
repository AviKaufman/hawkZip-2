#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h> // For uintptr_t
#include <emmintrin.h>
#include <omp.h>
#include <immintrin.h> // For AVX intrinsics

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
    unsigned char*  cmpData,
    int*   absQuant,
    unsigned int*  signFlag,
    int*  fixedRate,
    unsigned int*   threadOfs,
    size_t nbEle,
    size_t*  cmpSize,
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
    
    // Precompute quantization factor for better performance
    const float inv_err_bound = 0.5f / errorBound;
    
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

        // Allocate qBuf on stack once per thread
        int qBuf[BLOCK_SIZE];

        // each block
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;

            int bs = g * BLOCK_SIZE;
            int be = bs + BLOCK_SIZE;
            if (be > (int)nbEle) be = nbEle;
            int len = be - bs;

            // 1) quantize into a small local buffer - UNROLLED BY 8
            int i = 0;
            const __m256 inv_err_vec = _mm256_set1_ps(inv_err_bound);
            const __m256 half_vec    = _mm256_set1_ps(0.5f);
            
            // Process blocks of 8 elements at a time
            for (; i <= len - 8; i += 8) {
                const float* ptr = oriData + bs + i;
                __builtin_prefetch(ptr + 8, 0, 1);
            
                // load 8 floats, scale, add 0.5, then convert to int
                __m256  data    = _mm256_loadu_ps(ptr);
                __m256  scaled  = _mm256_mul_ps(data, inv_err_vec);
                __m256  shifted = _mm256_add_ps(scaled, half_vec);
                __m256i qi      = _mm256_cvtps_epi32(shifted);
            
                // store back into qBuf
                _mm256_storeu_si256((__m256i*)(qBuf + i), qi);
            }
            
            // Handle remaining elements with original method
            for (; i < len; i++) {
                float r = oriData[bs + i] * inv_err_bound;
                int sign = (r < -0.5f);
                qBuf[i] = (int)(r + 0.5f) - sign;
            }

            // 2) delta‑encode Q into absQuant[], build new signFlag
            unsigned int sflag = 0;
            int max_q = 0;

            // Fast path for first element
            int delta0 = qBuf[0];
            int a0 = delta0 < 0 ? -delta0 : delta0;
            absQuant[bs] = a0;
            
            // Use bit operations instead of modulo
            int bit_position = BLOCK_SIZE - 1 - (bs & BLOCK_MASK);
            sflag |= (unsigned int)(delta0 < 0) << bit_position;
            
            if (a0 > max_q) max_q = a0;
            int prev = qBuf[0];

            // Delta encode the rest with unrolling where possible
            i = 1;
            // Process blocks of 4 elements at a time for delta encoding
            for (; i <= len - 4; i += 4) {
                // Element 1
                int d1 = qBuf[i] - prev;
                int ad1 = d1 < 0 ? -d1 : d1;
                absQuant[bs + i] = ad1;
                bit_position = BLOCK_SIZE - 1 - ((bs + i) & BLOCK_MASK);
                sflag |= (unsigned int)(d1 < 0) << bit_position;
                if (ad1 > max_q) max_q = ad1;
                
                // Element 2
                int d2 = qBuf[i+1] - qBuf[i];
                int ad2 = d2 < 0 ? -d2 : d2;
                absQuant[bs + i + 1] = ad2;
                bit_position = BLOCK_SIZE - 1 - ((bs + i + 1) & BLOCK_MASK);
                sflag |= (unsigned int)(d2 < 0) << bit_position;
                if (ad2 > max_q) max_q = ad2;
                
                // Element 3
                int d3 = qBuf[i+2] - qBuf[i+1];
                int ad3 = d3 < 0 ? -d3 : d3;
                absQuant[bs + i + 2] = ad3;
                bit_position = BLOCK_SIZE - 1 - ((bs + i + 2) & BLOCK_MASK);
                sflag |= (unsigned int)(d3 < 0) << bit_position;
                if (ad3 > max_q) max_q = ad3;
                
                // Element 4
                int d4 = qBuf[i+3] - qBuf[i+2];
                int ad4 = d4 < 0 ? -d4 : d4;
                absQuant[bs + i + 3] = ad4;
                bit_position = BLOCK_SIZE - 1 - ((bs + i + 3) & BLOCK_MASK);
                sflag |= (unsigned int)(d4 < 0) << bit_position;
                if (ad4 > max_q) max_q = ad4;
                
                prev = qBuf[i+3];
            }

            // Handle remaining elements
            for (; i < len; i++) {
                int d = qBuf[i] - prev;
                int ad = d < 0 ? -d : d;
                absQuant[bs + i] = ad;
                
                bit_position = BLOCK_SIZE - 1 - ((bs + i) & BLOCK_MASK);
                sflag |= (unsigned int)(d < 0) << bit_position;
                
                if (ad > max_q) max_q = ad;
                prev = qBuf[i];
            }

            signFlag[g] = sflag;

            // 3) determine bit‑width using fast intrinsic
            int rate = max_q ? ((int)(sizeof(int)*8) - __builtin_clz(max_q)) : 0;
            fixedRate[g] = rate;
            cmpData[g] = (unsigned char)rate;

            if (rate)
                local_ofs += (BLOCK_SIZE + rate * BLOCK_SIZE) / 8;
        }

        threadOfs[tid] = local_ofs;
        #pragma omp barrier

        // prefix-sum to compute write offset (unmodified)
        unsigned int offs = 0;
        for (int t = 0; t < tid; t++)
            offs += threadOfs[t];
        unsigned int write_ptr = offs + totalBlocks;

        // 4) bit‑plane pack the deltas - improved with prefetching and better memory access
        for (int b = 0; b < perThread; b++) {
            int g = startBlock + b;
            if (g >= totalBlocks) break;
            int rate = fixedRate[g];
            if (!rate) continue;

            unsigned int sflag = signFlag[g];
            // write 4‑byte sign flag (optimized byte order write)
            cmpData[write_ptr++] = (sflag >> 24) & 0xFF;
            cmpData[write_ptr++] = (sflag >> 16) & 0xFF;
            cmpData[write_ptr++] = (sflag >> 8) & 0xFF;
            cmpData[write_ptr++] = sflag & 0xFF;

            // write bitplanes of absQuant[bs..be) - prefetch to improve memory access
            int bs = g * BLOCK_SIZE;
            int be = bs + BLOCK_SIZE;
            if (be > (int)nbEle) be = nbEle;
            
            // Prefetch absQuant data for better cache performance
            for (int i = bs; i < be; i += 16) {
                __builtin_prefetch(absQuant + i, 0, 3);
            }
            
            for (int bit = 0; bit < rate; bit++) {
                const unsigned int mask = 1U << bit;
                for (int chunk = 0; chunk < BLOCK_SIZE; chunk += 8) {
                    unsigned char packed = 0;
                    int limit = ((bs + chunk + 8) > be) ? (be - (bs + chunk)) : 8;
                    
                    // Fast path for full chunks (most common case)
                    if (limit == 8) {
                        for (int k = 0; k < 8; k++) {
                            packed |= ((absQuant[bs + chunk + k] & mask) >> bit) << (7 - k);
                        }
                    } else {
                        // Partial chunk (end of block)
                        for (int k = 0; k < limit; k++) {
                            packed |= ((absQuant[bs + chunk + k] & mask) >> bit) << (7 - k);
                        }
                    }
                    cmpData[write_ptr++] = packed;
                }
            }
        }

        // last thread writes total compressed size (unmodified)
        if (tid == THREAD_COUNT - 1) {
            unsigned int sum = 0;
            for (int t = 0; t < THREAD_COUNT; t++)
                sum += threadOfs[t];
            *cmpSize = sum + totalBlocks;
        }
    }
}


void hawkZip_decompress_kernel(
    float* decData,
    const unsigned char* cmpData, // Added const
    int* absQuant,
    int* fixedRate,
    unsigned int* threadOfs,
    const size_t nbEle,          // Added const
    const float errorBound)      // Added const
{
    if (nbEle == 0) return;

    const int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    omp_set_num_threads(THREAD_COUNT);

    // Optimization 1: Precompute dequantization factor
    const float dequant_factor = errorBound * 2.0f;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        const int perThread  = (totalBlocks + THREAD_COUNT - 1) / THREAD_COUNT;
        const int startBlock = tid * perThread;
        unsigned int local_ofs = 0;

        // 1) read per‑block rates (Keep original logic)
        for (int b = 0; b < perThread; ++b) {
            const int g = startBlock + b;
            if (g >= totalBlocks) break;
            const int rate = (int)cmpData[g];
            fixedRate[g] = rate;
            if (rate) {
                // *** KEEP THE ORIGINAL OFFSET CALCULATION ***
                local_ofs += (BLOCK_SIZE + rate * BLOCK_SIZE) / 8;
            }
        }
        threadOfs[tid] = local_ofs;
        #pragma omp barrier // Barrier essential for prefix sum

        // 2) prefix‑sum to get read offset (Keep original logic)
        unsigned int offs = 0;
        for (int t = 0; t < tid; ++t) {
            offs += threadOfs[t];
        }
        unsigned int read_ptr = offs + totalBlocks;

        // 3) unpack (Keep most original logic)
        for (int b = 0; b < perThread; ++b) {
            const int g = startBlock + b;
            if (g >= totalBlocks) break;

            const int rate = fixedRate[g];
            if (!rate) continue;

            // read sign flag (Keep original logic)
            unsigned int sflag = 0;
            for (int byte = 3; byte >= 0; --byte) {
                 sflag |= ((unsigned int)cmpData[read_ptr++]) << (8 * byte);
            }

            const int bs = g * BLOCK_SIZE;
            const int be = ((bs + BLOCK_SIZE) > (int)nbEle) ? (int)nbEle : (bs + BLOCK_SIZE);
            const int block_elem_count = be - bs;

            if (block_elem_count <= 0) continue;

            // Optimization 2: Use memset to clear absQuant for the current block
            memset(absQuant + bs, 0, (size_t)block_elem_count * sizeof(int));

            // read bitplanes (Keep original logic)
            for (int bit = 0; bit < rate; ++bit) {
                for (int chunk_offset = 0; chunk_offset < block_elem_count; chunk_offset += 8) {
                     const unsigned char packed = cmpData[read_ptr++];
                     const int limit = ((chunk_offset + 8) > block_elem_count) ? (block_elem_count - chunk_offset) : 8;
                     const int current_block_elem_start = bs + chunk_offset;
                     for (int k = 0; k < limit; ++k) {
                         // Original bit unpacking logic:
                         absQuant[current_block_elem_start + k] |=
                           ((packed >> (7 - k)) & 1) << bit;
                     }
                }
            }

            // 4) delta‑decode + dequantize into decData (Keep original logic, use precomputed factor)
            int prevQ = 0;
            for (int i = bs; i < be; ++i) {
                const int sign = (sflag >> (BLOCK_SIZE - 1 - (i % BLOCK_SIZE))) & 1;
                const int q_mag = absQuant[i];
                const int d = sign ? -q_mag : q_mag;

                if (i == bs) {
                    prevQ = d;
                } else {
                    prevQ += d;
                }
                // Use precomputed factor (Optimization 1 applied here)
                decData[i] = (float)prevQ * dequant_factor;
            }
        } // End loop over blocks (b)
    } // End parallel region
}