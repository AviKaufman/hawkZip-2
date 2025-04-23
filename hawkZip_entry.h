#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <zlib.h>
#include "hawkZip_compressor.h"

void hawkZip_compress(
    float*         oriData,
    unsigned char* cmpData,
    size_t         nbEle,
    size_t*        cmpSize,
    float          errorBound)
{
    // Stage 1: existing delta + bit-plane compressor
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

    // Stage 2: DEFLATE (zlib) compress the bit-plane payload
    uLongf destLen = compressBound((uLong)origSize);
    unsigned char* defBuf = malloc(destLen);
    int zret = compress(
        defBuf, &destLen,
        cmpData, (uLong)origSize);
    if (zret != Z_OK) {
        fprintf(stderr, "zlib compress error: %d\n", zret);
        exit(1);
    }

    // 3) emit 8-byte header + DEFLATE data
    uint32_t* hdr = (uint32_t*)cmpData;
    hdr[0] = (uint32_t)origSize;
    hdr[1] = (uint32_t)destLen;
    memcpy(cmpData + 8, defBuf, destLen);
    free(defBuf);

    *cmpSize = 8 + destLen;

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

    // 2) DEFLATE (zlib) decompress into planeBuf
    unsigned char* planeBuf = malloc((size_t)origSize);
    uLongf outLen = (uLongf)origSize;
    int zret = uncompress(
        planeBuf, &outLen,
        src, (uLong)cSize);
    if (zret != Z_OK || outLen != (uLongf)origSize) {
        fprintf(stderr, "zlib uncompress error: %d (got %lu)\n", zret, outLen);
        exit(1);
    }

    // 3) existing delta-unpack decompressor
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
