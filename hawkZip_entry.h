// hawkZip_entry.h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <zstd.h>
#include <stdint.h>
#include <float.h> // For DBL_MAX used in stats calculation
#include "hawkZip_compressor.h"

// Define block size based on kernel logic (usually 32)
#define BLOCK_SIZE 32
// Define number of benchmark runs
#define NUM_RUNS 100

/**
 * @brief Calculates minimum, maximum, and average values from an array of doubles.
 * @param values Array of double values.
 * @param count Number of elements in the array.
 * @param min_val Pointer to store the minimum value.
 * @param max_val Pointer to store the maximum value.
 * @param avg_val Pointer to store the average value.
 */
static void calculate_stats(const double values[], int count, double *min_val, double *max_val, double *avg_val)
{
    if (count <= 0)
    {
        *min_val = *max_val = *avg_val = 0.0;
        return;
    }

    *min_val = DBL_MAX;
    *max_val = -DBL_MAX; // Or 0.0 if values are non-negative
    double sum = 0.0;
    for (int i = 0; i < count; ++i)
    {
        sum += values[i];
        if (values[i] < *min_val)
            *min_val = values[i];
        if (values[i] > *max_val)
            *max_val = values[i];
    }
    *avg_val = sum / count;
}

void hawkZip_compress(float *oriData, unsigned char *cmpData, size_t nbEle, size_t *cmpSize, float errorBound)
{
    if (nbEle == 0)
    {
        *cmpSize = 0;
        return;
    }

    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate kernel buffers ONCE outside the loop
    int *absQuant = (int *)malloc(sizeof(int) * nbEle);
    unsigned int *signFlag = (unsigned int *)malloc(sizeof(unsigned int) * totalBlocks);
    int *fixedRate = (int *)malloc(sizeof(int) * totalBlocks);

    if (!absQuant || !signFlag || !fixedRate)
    {
        perror("Memory allocation failed for HawkZip kernel buffers");
        free(absQuant); // Free any that succeeded
        free(signFlag);
        free(fixedRate);
        *cmpSize = 0;
        return;
    }

    // Arrays to store timing results for each run
    double compress_throughputs[NUM_RUNS];
    size_t final_cmpSize = 0; // Capture size from the last valid run
    int valid_runs = 0;       // Count successful runs

    printf("Running compression benchmark (%d runs)...\n", NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        size_t hawkZipOutSize = 0;
        double timerCMP_kernel_start, timerCMP_kernel_end;
        double timerCMP_zstd_start, timerCMP_zstd_end;
        double total_compress_time = 0.0;
        size_t current_run_cmpSize = 0;

        // 1. --- Run original HawkZip kernel ---
        timerCMP_kernel_start = omp_get_wtime();
        hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate,
                                nbEle, &hawkZipOutSize, errorBound);
        timerCMP_kernel_end = omp_get_wtime();

        if (hawkZipOutSize == 0)
        {
            fprintf(stderr, "Run %d: HawkZip kernel compression failed.\n", run + 1);
            continue; // Skip to next run
        }

        // 2. --- ZSTD compress the kernel output ---
        timerCMP_zstd_start = omp_get_wtime();
        size_t zstdBound = ZSTD_compressBound(hawkZipOutSize);
        unsigned char *zBuf = malloc(zstdBound);
        if (!zBuf)
        {
            perror("Run %d: Memory allocation failed for ZSTD buffer");
            // Cannot reliably continue this run
            continue;
        }

        int zstdLevel = 3; // Example level
        size_t cSize = ZSTD_compress(zBuf, zstdBound, cmpData, hawkZipOutSize, zstdLevel);
        timerCMP_zstd_end = omp_get_wtime();

        if (ZSTD_isError(cSize))
        {
            fprintf(stderr, "Run %d: ZSTD compression error: %s\n", run + 1, ZSTD_getErrorName(cSize));
            free(zBuf);
            continue; // Skip to next run
        }

        // 3. --- Write header + ZSTD data back into cmpData ---
        uint32_t *hdr = (uint32_t *)cmpData;
        size_t headerSize = 2 * sizeof(uint32_t);
        current_run_cmpSize = headerSize + cSize;

        // Optional Warning: Check if likely to overflow buffer provided by main
        // This check is heuristic as we don't know the actual buffer capacity
        if (current_run_cmpSize > hawkZipOutSize && run == 0)
        {
            printf("Warning: Final compressed size (%zu) might exceed initial buffer estimate implied by kernel output size (%zu).\n", current_run_cmpSize, hawkZipOutSize);
        }

        hdr[0] = (uint32_t)hawkZipOutSize;
        hdr[1] = (uint32_t)cSize;
        memcpy(cmpData + headerSize, zBuf, cSize);
        free(zBuf);

        // --- Calculate & Store Throughput for this run ---
        total_compress_time = (timerCMP_kernel_end - timerCMP_kernel_start) + (timerCMP_zstd_end - timerCMP_zstd_start);
        if (total_compress_time > 1e-9) // Avoid division by zero or near-zero
        {
            compress_throughputs[valid_runs] = (nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / total_compress_time;
            final_cmpSize = current_run_cmpSize; // Store size from the last successful run
            valid_runs++;
        }
        else
        {
            fprintf(stderr, "Run %d: Compression time too small to measure throughput.\n", run + 1);
        }
    } // End benchmark loop

    printf("Compression benchmark finished (%d valid runs).\n", valid_runs);

    // --- Calculate and Print Stats ---
    if (valid_runs > 0)
    {
        double min_tp, max_tp, avg_tp;
        calculate_stats(compress_throughputs, valid_runs, &min_tp, &max_tp, &avg_tp);

        // Update the output cmpSize parameter with the size from the last valid run
        *cmpSize = final_cmpSize;

        // Print metrics
        printf("hawkZip   compression ratio:      %f (based on size %zu bytes)\n",
               (float)(sizeof(float) * nbEle) / (float)(*cmpSize), *cmpSize);
        printf("hawkZip   compression throughput: avg=%.4f GB/s (min=%.4f, max=%.4f)\n",
               avg_tp, min_tp, max_tp);
    }
    else
    {
        fprintf(stderr, "No valid compression runs completed.\n");
        *cmpSize = 0; // Indicate failure if no runs worked
    }

    // --- Free kernel buffers ---
    free(absQuant);
    free(signFlag);
    free(fixedRate);
}

void hawkZip_decompress(float *decData, unsigned char *cmpData, size_t nbEle, float errorBound)
{
    if (nbEle == 0 || cmpData == NULL)
    {
        fprintf(stderr, "Decompression Error: Invalid input (nbEle=%zu, cmpData=%p).\n", nbEle, (void *)cmpData);
        memset(decData, 0, sizeof(float) * nbEle); // Ensure output is cleared
        return;
    }

    // Allocate kernel buffers ONCE outside the loop
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *absQuant = (int *)malloc(sizeof(int) * nbEle);
    int *fixedRate = (int *)malloc(sizeof(int) * totalBlocks);

    if (!absQuant || !fixedRate)
    {
        perror("Memory allocation failed for HawkZip kernel buffers");
        free(absQuant);
        free(fixedRate);
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }

    // Arrays to store timing results
    double decompress_throughputs[NUM_RUNS];
    int valid_runs = 0;

    printf("Running decompression benchmark (%d runs)...\n", NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        double timerDEC_zstd_start, timerDEC_zstd_end;
        double timerDEC_kernel_start, timerDEC_kernel_end;
        double total_decompress_time = 0.0;
        unsigned char *planeBuf = NULL; // Declare here for scope

        // 1. --- Read header from cmpData ---
        size_t headerSize = 2 * sizeof(uint32_t);
        uint32_t origSize = ((uint32_t *)cmpData)[0];
        uint32_t cSize = ((uint32_t *)cmpData)[1];
        unsigned char *src = cmpData + headerSize;

        if (origSize == 0 || cSize == 0)
        {
            // This error should likely only happen once if input is bad
            if (run == 0)
                fprintf(stderr, "Decompression Error: Header contains zero sizes (orig=%u, cSize=%u).\n", origSize, cSize);
            continue;
        }

        // 2. --- ZSTD decompress into a temporary buffer ---
        timerDEC_zstd_start = omp_get_wtime();
        planeBuf = malloc(origSize); // Allocate inside loop
        if (!planeBuf)
        {
            perror("Run %d: Memory allocation failed for ZSTD decompression buffer");
            continue;
        }
        size_t dSize = ZSTD_decompress(planeBuf, origSize, src, cSize);
        timerDEC_zstd_end = omp_get_wtime();

        if (ZSTD_isError(dSize))
        {
            fprintf(stderr, "Run %d: ZSTD decompression error: %s\n", run + 1, ZSTD_getErrorName(dSize));
            free(planeBuf);
            continue;
        }
        if (dSize != origSize)
        {
            fprintf(stderr, "Run %d: ZSTD decompression size mismatch: Expected %u, got %zu\n", run + 1, origSize, dSize);
            free(planeBuf);
            continue;
        }

        // 3. --- Run original HawkZip kernel on decompressed data ---
        // Memset buffers modified by kernel inside the loop
        // Kernel assumes zeroed input sometimes
        memset(absQuant, 0, sizeof(int) * nbEle);
        // Clear output buffer for this run
        memset(decData, 0, sizeof(float) * nbEle);

        timerDEC_kernel_start = omp_get_wtime();
        hawkZip_decompress_kernel(decData, planeBuf, absQuant, fixedRate,
                                  nbEle, errorBound);
        timerDEC_kernel_end = omp_get_wtime();

        // --- Calculate & Store Throughput ---
        total_decompress_time = (timerDEC_zstd_end - timerDEC_zstd_start) + (timerDEC_kernel_end - timerDEC_kernel_start);
        if (total_decompress_time > 1e-9) // Avoiding divide by zero error
        {
            decompress_throughputs[valid_runs] = (nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / total_decompress_time;
            valid_runs++;
        }
        else
        {
            fprintf(stderr, "Run %d: Decompression time too small to measure throughput.\n", run + 1);
        }

        // --- Free per-run buffer ---
        free(planeBuf); // Free buffer allocated in this iteration
    }

    printf("Decompression benchmark finished (%d valid runs).\n", valid_runs);

    // --- Calculate and Print Stats ---
    if (valid_runs > 0)
    {
        double min_tp, max_tp, avg_tp;
        calculate_stats(decompress_throughputs, valid_runs, &min_tp, &max_tp, &avg_tp);

        // Print metrics
        printf("hawkZip decompression throughput: avg=%.4f GB/s (min=%.4f, max=%.4f)\n",
               avg_tp, min_tp, max_tp);
    }
    else
    {
        fprintf(stderr, "No valid decompression runs completed.\n");
        // Output decData remains zeroed from last attempt or initial state
    }

    // --- Free buffers allocated once ---
    free(absQuant);
    free(fixedRate);
}