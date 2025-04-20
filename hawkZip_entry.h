#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <zstd.h>
#include "hawkZip_compressor.h"

void hawkZip_compress(
    float*         oriData,
    unsigned char* cmpData,
    size_t         nbEle,
    size_t*        cmpSize,
    float          errorBound)
{
    // existing delta compressor
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int*           absQuant  = malloc(sizeof(int)           * nbEle);
    unsigned int*  signFlag  = malloc(sizeof(unsigned int) * totalBlocks);
    int*           fixedRate = malloc(sizeof(int)           * totalBlocks);
    unsigned int*  threadOfs = malloc(sizeof(unsigned int) * THREAD_COUNT);

    double t0 = omp_get_wtime();
    hawkZip_compress_kernel(
        oriData, cmpData,
        absQuant, signFlag,
        fixedRate, threadOfs,
        nbEle, cmpSize,
        errorBound);
    double t1 = omp_get_wtime();

    int origSize = (int)*cmpSize;

    // 2) Zstd compress the bit‑plane payload
    unsigned char* zBuf = malloc(ZSTD_compressBound(origSize));
    size_t cSize = ZSTD_compress(
        zBuf, ZSTD_compressBound(origSize),
        cmpData, origSize,
        42 /* speed vs. ratio level */);

    // 3) emit 8‑byte header + Zstd data
    uint32_t* hdr = (uint32_t*)cmpData;
    hdr[0] = (uint32_t)origSize;
    hdr[1] = (uint32_t)cSize;
    memcpy(cmpData + 8, zBuf, cSize);
    free(zBuf);

    *cmpSize = 8 + cSize;

    // 4) print metrics
    printf("hawkZip   compression ratio:      %f\n",
           (float)(sizeof(float)*nbEle)/(float)(*cmpSize));
    printf("hawkZip   compression throughput: %f GB/s\n",
           (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(t1-t0));

    free(absQuant);
    free(signFlag);
    free(fixedRate);
    free(threadOfs);
}

void hawkZip_decompress(
    float*         decData,
    unsigned char* cmpData,
    size_t         nbEle,
    float          errorBound)
{
    // compute totalBlocks here
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1) peel off header
    uint32_t origSize = ((uint32_t*)cmpData)[0];
    uint32_t cSize    = ((uint32_t*)cmpData)[1];
    unsigned char* src = cmpData + 8;

    // 2) Zstd decompress into planeBuf
    unsigned char* planeBuf = malloc(origSize);
    size_t dSize = ZSTD_decompress(
        planeBuf, origSize,
        src, cSize);
    if (dSize != origSize) {
        fprintf(stderr, "ZSTD decompression error: %zu != %u\n",
                dSize, origSize);
        exit(1);
    }

    // 3) your existing delta‑unpack decompressor
    int*           absQuant  = calloc(nbEle, sizeof(int));
    int*           fixedRate = malloc(sizeof(int) * totalBlocks);
    unsigned int*  threadOfs = malloc(sizeof(unsigned int) * THREAD_COUNT);

    double t0 = omp_get_wtime();
    hawkZip_decompress_kernel(
        decData, planeBuf,
        absQuant, fixedRate, threadOfs,
        nbEle, errorBound);
    double t1 = omp_get_wtime();

    printf("hawkZip decompression throughput: %f GB/s\n",
           (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(t1-t0));

    free(planeBuf);
    free(absQuant);
    free(fixedRate);
    free(threadOfs);
}