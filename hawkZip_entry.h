// hawkZip_entry.h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <zstd.h>
#include <stdint.h>
#include "hawkZip_compressor.h"

// Define block size based on kernel logic (usually 32)
#define BLOCK_SIZE 32

void hawkZip_compress(float *oriData, unsigned char *cmpData, size_t nbEle, size_t *cmpSize, float errorBound)
{
    // 1. --- Run original HawkZip kernel ---

    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate buffers for the kernel
    int *absQuant = (int *)malloc(sizeof(int) * nbEle);
    unsigned int *signFlag = (unsigned int *)malloc(sizeof(unsigned int) * totalBlocks);
    int *fixedRate = (int *)malloc(sizeof(int) * totalBlocks);

    if (!absQuant || !signFlag || !fixedRate)
    {
        perror("Memory allocation failed for HawkZip kernel buffers");
        free(absQuant);
        free(signFlag);
        free(fixedRate);
        *cmpSize = 0;
        return;
    }

    size_t hawkZipOutSize = 0;
    double timerCMP_kernel_start, timerCMP_kernel_end;
    double timerCMP_zstd_start, timerCMP_zstd_end;
    double total_compress_time;

    timerCMP_kernel_start = omp_get_wtime();
    // Call kernel, note threadOfs is removed from kernel signature
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate,
                            nbEle, &hawkZipOutSize, errorBound);
    timerCMP_kernel_end = omp_get_wtime();

    if (hawkZipOutSize == 0)
    {
        fprintf(stderr, "HawkZip kernel compression returned size 0, indicating failure.\n");
        free(absQuant);
        free(signFlag);
        free(fixedRate);
        *cmpSize = 0;
        return;
    }

    // 2. --- ZSTD compress the kernel output ---

    timerCMP_zstd_start = omp_get_wtime();
    size_t zstdBound = ZSTD_compressBound(hawkZipOutSize);
    unsigned char *zBuf = malloc(zstdBound);
    if (!zBuf)
    {
        perror("Memory allocation failed for ZSTD buffer");
        free(absQuant);
        free(signFlag);
        free(fixedRate);
        *cmpSize = 0;
        return;
    }

    int zstdLevel = 3;
    size_t cSize = ZSTD_compress(zBuf, zstdBound, cmpData, hawkZipOutSize, zstdLevel);
    timerCMP_zstd_end = omp_get_wtime();

    if (ZSTD_isError(cSize))
    {
        fprintf(stderr, "ZSTD compression error: %s\n", ZSTD_getErrorName(cSize));
        free(zBuf);
        free(absQuant);
        free(signFlag);
        free(fixedRate);
        *cmpSize = 0;
        return;
    }

    // 3. --- Write header + ZSTD data back into cmpData ---

    uint32_t *hdr = (uint32_t *)cmpData;
    size_t headerSize = 2 * sizeof(uint32_t);

    if (headerSize + cSize > hawkZipOutSize)
    {
        printf("Warning: Final compressed size (%zu) might exceed initial buffer estimate implied by kernel output size (%zu).\n", headerSize + cSize, hawkZipOutSize);
    }

    hdr[0] = (uint32_t)hawkZipOutSize;
    hdr[1] = (uint32_t)cSize;
    memcpy(cmpData + headerSize, zBuf, cSize);
    free(zBuf);

    *cmpSize = headerSize + cSize;

    // --- Calculate Total Time ---
    total_compress_time = (timerCMP_kernel_end - timerCMP_kernel_start) + (timerCMP_zstd_end - timerCMP_zstd_start);

    // --- Print metrics in the order yafan gave us ---
    // 1. Compression Ratio
    printf("hawkZip   compression ratio:      %f\n", (float)(sizeof(float) * nbEle) / (float)(*cmpSize));
    // 2. Compression Throughput (using total time)
    printf("hawkZip   compression throughput: %f GB/s\n", (nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / total_compress_time);

    // --- Free kernel buffers ---
    free(absQuant);
    free(signFlag);
    free(fixedRate);
}

void hawkZip_decompress(float *decData, unsigned char *cmpData, size_t nbEle, float errorBound)
{
    double timerDEC_zstd_start, timerDEC_zstd_end;
    double timerDEC_kernel_start, timerDEC_kernel_end;
    double total_decompress_time;

    // 1. --- Read header from cmpData ---
    size_t headerSize = 2 * sizeof(uint32_t);
    if (cmpData == NULL)
    {
        fprintf(stderr, "Decompression Error: Input cmpData is NULL.\n");
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }

    uint32_t origSize = ((uint32_t *)cmpData)[0];
    uint32_t cSize = ((uint32_t *)cmpData)[1];
    unsigned char *src = cmpData + headerSize;

    if (origSize == 0 || cSize == 0)
    {
        fprintf(stderr, "Decompression Error: Header contains zero sizes (orig=%u, cSize=%u).\n", origSize, cSize);
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }

    // 2. --- ZSTD decompress into a temporary buffer ---

    timerDEC_zstd_start = omp_get_wtime();
    unsigned char *planeBuf = malloc(origSize);
    if (!planeBuf)
    {
        perror("Memory allocation failed for ZSTD decompression buffer (planeBuf)");
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }
    size_t dSize = ZSTD_decompress(planeBuf, origSize, src, cSize);
    timerDEC_zstd_end = omp_get_wtime();

    if (ZSTD_isError(dSize))
    {
        fprintf(stderr, "ZSTD decompression error: %s\n", ZSTD_getErrorName(dSize));
        free(planeBuf);
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }
    if (dSize != origSize)
    {
        fprintf(stderr, "ZSTD decompression size mismatch: Expected %u, got %zu\n", origSize, dSize);
        free(planeBuf);
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }

    // 3. --- Run original HawkZip kernel on decompressed data ---

    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *absQuant = (int *)malloc(sizeof(int) * nbEle);
    int *fixedRate = (int *)malloc(sizeof(int) * totalBlocks);

    if (!absQuant || !fixedRate)
    {
        perror("Memory allocation failed for HawkZip kernel buffers");
        free(planeBuf);
        free(absQuant);
        free(fixedRate);
        memset(decData, 0, sizeof(float) * nbEle);
        return;
    }

    memset(absQuant, 0, sizeof(int) * nbEle);
    memset(decData, 0, sizeof(float) * nbEle);

    timerDEC_kernel_start = omp_get_wtime();
    // Call kernel, note threadOfs is removed from kernel signature
    hawkZip_decompress_kernel(decData, planeBuf, absQuant, fixedRate,
                              nbEle, errorBound);
    timerDEC_kernel_end = omp_get_wtime();

    // --- Calculate Total Time ---
    total_decompress_time = (timerDEC_zstd_end - timerDEC_zstd_start) + (timerDEC_kernel_end - timerDEC_kernel_start);

    // --- Print metrics in the desired order yafan gave us ---
    // 3. Decompression Throughput (using total time)
    printf("hawkZip decompression throughput: %f GB/s\n", (nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / total_decompress_time);

    // --- Free buffers ---
    free(planeBuf);
    free(absQuant);
    free(fixedRate);

    // Pray there are no unbounded errors.
}