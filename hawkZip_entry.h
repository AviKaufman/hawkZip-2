// hawkZip_entry.h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <blosc.h>
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
    

    int origSize = (int)*cmpSize;

    // Stage 2: Blosc compress the bit-plane payload
    blosc_init();
    size_t  destCap  = (size_t)origSize + BLOSC_MAX_OVERHEAD;
    unsigned char* bloscBuf = malloc(destCap);
    int cSize = blosc_compress(
        /*clevel=*/5,
        /*doshuffle=*/BLOSC_SHUFFLE,
        /*typesize=*/1,
        /*nbytes=*/(size_t)origSize,
        /*src=*/cmpData,
        /*dest=*/bloscBuf,
        /*destsize=*/destCap
    );
    if (cSize <= 0) {
        fprintf(stderr, "Blosc compress error: %d\n", cSize);
        exit(1);
    }
    blosc_destroy();

    // 3) emit 8-byte header + Blosc data
    uint32_t* hdr = (uint32_t*)cmpData;
    hdr[0] = (uint32_t)origSize;
    hdr[1] = (uint32_t)cSize;
    memcpy(cmpData + 8, bloscBuf, (size_t)cSize);
    free(bloscBuf);

    *cmpSize = 8 + cSize;
    double t1 = omp_get_wtime();
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
    int totalBlocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double t0 = omp_get_wtime();
    // 1) peel off header
    uint32_t origSize = ((uint32_t*)cmpData)[0];
    uint32_t cSize    = ((uint32_t*)cmpData)[1];
    unsigned char* src = cmpData + 8;

    // 2) Blosc decompress into planeBuf
    blosc_init();
    unsigned char* planeBuf = malloc((size_t)origSize);
    long dSize = blosc_decompress(
        /*src=*/src,
        /*dest=*/planeBuf,
        /*destsize=*/(size_t)origSize
    );
    if (dSize <= 0) {
        fprintf(stderr, "Blosc decompress error: %ld\n", dSize);
        exit(1);
    }
    blosc_destroy();

    // 3) existing delta-unpack decompressor
    int*           absQuant  = calloc(nbEle, sizeof(int));
    int*           fixedRate = malloc(sizeof(int) * totalBlocks);
    unsigned int*  threadOfs = malloc(sizeof(unsigned int) * THREAD_COUNT);

    
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
