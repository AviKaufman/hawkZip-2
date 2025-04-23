#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <lz4.h>
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

    // 2) LZ4 compress the bit-plane payload
    int maxOut = LZ4_compressBound(origSize);
    unsigned char* lzBuf = malloc((size_t)maxOut);
    int cSize = LZ4_compress_default(
        (const char*)cmpData, (char*)lzBuf,
        origSize, maxOut);
    if (cSize <= 0) {
        fprintf(stderr, "LZ4 compression error: %d\n", cSize);
        exit(1);
    }

    // 3) emit 8‑byte header + Zstd data
    uint32_t* hdr = (uint32_t*)cmpData;
    hdr[0] = (uint32_t)origSize;
    hdr[1] = (uint32_t)cSize;
    memcpy(cmpData + 8, lzBuf, (size_t)cSize);
    free(lzBuf);

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

    // 2) LZ4 decompress into planeBuf
    unsigned char* planeBuf = malloc((size_t)origSize);
    int dSize = LZ4_decompress_safe(
        (const char*)src, (char*)planeBuf,
        cSize, origSize);
    if (dSize < 0) {
        fprintf(stderr, "LZ4 decompression error: %d\n", dSize);
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