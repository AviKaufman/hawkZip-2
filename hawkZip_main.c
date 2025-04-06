#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hawkZip_entry.h"
#include "hawkZip_utility.h"

int main(int argc, char* argv[])
{
    // Read input arguments.
    char srcFilePath[640] = "";
    float errorBound = 0.0f;
    char cmpFilePath[640] = "";
    char decFilePath[640] = "";

    // Analyze input arguments.
    if (argc < 5)
    {
        printf("hawkZip: a lossy compression tool for floating-point data in UIowa:CS:4700\n");
        printf("Usage:\n");
        printf("  %s -i [srcFilePath] -e [errorBound] [-x cmpFilePath] [-o decFilePath]\n", argv[0]);
        printf("Options:\n");
        printf("  -i [srcFilePath]  Required. Input file path of original data\n");
        printf("  -e [errorBound]   Required. Relative error bound\n");
        printf("  -x [cmpFilePath]  Optional. Output file path of compressed data\n");
        printf("  -o [decFilePath]  Optional. Output file path of decompressed data\n");
        printf("Example:\n");
        printf("  %s -i original_file.f32 -e 1e-2\n", argv[0]);
        printf("  %s -i original_file.f32 -e 1e-4 -x comprssed.bin\n", argv[0]);
        printf("  %s -i original_file.f32 -e 0.01 -o decompressed.bin\n", argv[0]);
        printf("  %s -i original_file.f32 -e 0.35 -x compressed.bin -o decompressed.bin\n", argv[0]);
        return 1;
    }

    // Loop over all command-line arguments
    for (int i = 1; i < argc; i++)
    {
        // Check for -i (required)
        if ((strcmp(argv[i], "-i") == 0) && (i + 1 < argc))
        {
            strcpy(srcFilePath, argv[++i]);
        }
        // Check for -e (required)
        else if ((strcmp(argv[i], "-e") == 0) && (i + 1 < argc))
        {
            errorBound = (float)atof(argv[++i]);
        }
        // Check for -x (optional)
        else if ((strcmp(argv[i], "-x") == 0) && (i + 1 < argc))
        {
            strcpy(cmpFilePath, argv[++i]);
        }
        // Check for -o (optional)
        else if ((strcmp(argv[i], "-o") == 0) && (i + 1 < argc))
        {
            strcpy(decFilePath, argv[++i]);
        }
        // Unrecognized argument or missing its value
        else
        {
            printf("Unrecognized or incomplete option '%s'\n", argv[i]);
            printf("Usage:\n");
            printf("  %s -i [srcFilePath] -e [errorBound] [-x cmpFilePath] [-o decFilePath]\n", argv[0]);
            return 1;
        }
    }

    // Basic validation for required fields
    if (srcFilePath[0] == '\0' || errorBound == 0.0f)
    {
        printf("Error: missing required arguments!\n");
        printf("  -i [srcFilePath] and -e [errorBound] must be provided.\n");
        printf("Usage:\n");
        printf("  %s -i [srcFilePath] -e [errorBound] [-x cmpFilePath] [-o decFilePath]\n", argv[0]);
        return 1;
    }

    // Define variables.
    float* oriData = NULL;  // original data
    unsigned char* cmpData = NULL;  // compressed data
    float* decData = NULL;  // decomprssed data

    // Initialize variables.
    size_t nbEle = 0;
    size_t cmpSize = 0;
    int status=0;
    oriData = readFloatData_Yafan(srcFilePath, &nbEle, &status);
    cmpData = (unsigned char*)calloc(4*nbEle, sizeof(unsigned char));
    decData = (float*)calloc(nbEle, sizeof(float));

    // Get value-range-based relative error bound.
    float max_val = oriData[0];
    float min_val = oriData[0];
    for(size_t i=0; i<nbEle; i++) {
        if(oriData[i]>max_val)
            max_val = oriData[i];
        else if(oriData[i]<min_val)
            min_val = oriData[i];
    }
    errorBound = errorBound * (max_val - min_val);
    
    // Compression.
    hawkZip_compress(oriData, cmpData, nbEle, &cmpSize, errorBound);

    // Decompression.
    hawkZip_decompress(decData, cmpData, nbEle, errorBound);

    // Error check
    printf("\n");
    printf("Now error checking ...\n");
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        // Avoid rounding issue in float-point calculation, so use errorbound*1.1
        if(fabs(oriData[i]-decData[i])>errorBound*1.1) 
        {
            // printf("%zu %f %f %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]));
            not_bound++;
        }
    }
    if(not_bound) printf("\033[0;31mError Check Fail!\033[0m %d unbounded data points.\n", not_bound);
    else printf("\033[0;32mError Check Success!\033[0m\n");

    // Write compressed or reconstructed/decompressed data if needed.
    if (cmpFilePath[0] != '\0')
    {
        printf("\n");
        printf("Writing compressed data to %s...\n", cmpFilePath);
        writeByteData_Yafan(cmpData, cmpSize, cmpFilePath, &status);
    }
    if (decFilePath[0] != '\0')
    {
        printf("\n");
        printf("Writing decompressed data to %s...\n", decFilePath);
        writeFloatData_inBytes_Yafan(decData, nbEle, decFilePath, &status);
    }

    free(oriData);
    free(cmpData);
    free(decData);

    return 0;
}

