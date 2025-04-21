#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
// #include <emmintrin.h> // commented out for now because it isn't compability with mac

// NUM_THREADS is defined outside this snippet now
// extern int NUM_THREADS; // Assuming NUM_THREADS is defined globally or passed differently

#define NUM_THREADS 8

#ifndef DIM_X
#define DIM_X 3600
#endif
#define BLOCK_SIZE 32 // maybe try changing block size to get better performance??

// 2D Lorenzo Predictor
static inline float lorenzo_pred(const float *d, size_t idx, size_t dim_x)
{
    if (idx < dim_x || idx % dim_x == 0)
        return 0.0f;
    size_t up = idx - dim_x, left = idx - 1, upleft = up - 1;
    return d[up] + d[left] - d[upleft];
}

// Max helper. Honestly, this is a waste of space, but whatever.
static inline int max_int(int a, int b) { return a > b ? a : b; }

// --- Compression Kernel with Wavefront & Block Parallel Passes ---
void hawkZip_compress_kernel(float *oriData, unsigned char *cmpData, int *absQuantResidual, unsigned int *signFlag, int *fixedRate,
                             size_t nbEle, size_t *cmpSize, float errorBound)
{
    if (nbEle == 0)
    {
        *cmpSize = 0;
        return;
    }

    // --- Setup ---
    const size_t dim_x = DIM_X;
    const size_t dim_y = (nbEle + dim_x - 1) / dim_x;
    const int total_blocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float recip_precision = 0.5f / errorBound;

    float *decLocal = (float *)calloc(nbEle, sizeof(float));
    if (!decLocal)
    {
        perror("calloc failed for decLocal");
        *cmpSize = 0;
        return;
    }

    omp_set_num_threads(NUM_THREADS); // Set threads for subsequent parallel regions

    // --- Pass 1: Wavefront Prediction, Quantization, Reconstruction ---
    for (size_t d = 0; d < dim_y + dim_x - 1; ++d)
    {
        size_t r_start = (d >= dim_x) ? (d - dim_x + 1) : 0;
        size_t r_end = (d < dim_y) ? d : (dim_y - 1);
#pragma omp parallel for schedule(static) // Parallelize WITHIN diagonal (appears to be less efficient than simple block parallelization). But it technically removes dependencies, so it stay for now.
        for (size_t r = r_start; r <= r_end; ++r)
        {
            size_t c = d - r;
            if (c >= dim_x)
                continue;
            size_t idx = r * dim_x + c;
            if (idx >= nbEle)
                continue;

            float prediction = lorenzo_pred(decLocal, idx, dim_x);
            float residual = oriData[idx] - prediction;
            float residual_recip = residual * recip_precision;
            int s = residual_recip >= -0.5f ? 0 : 1;
            int curr_quant_residual = (int)(residual_recip + 0.5f) - s;
            int abs_quant_res = abs(curr_quant_residual);
            absQuantResidual[idx] = abs_quant_res;
            float decompressed_residual = (float)curr_quant_residual * errorBound * 2.0f;
            decLocal[idx] = prediction + decompressed_residual;
        }
    } // End Wavefront Pass 1

// --- Pass 2: Metadata Calculation (Block Parallel - Block Assignment) ---
#pragma omp parallel for schedule(static) // Distribute BLOCKS among threads
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        int block_start_global = block_idx * BLOCK_SIZE;
        int block_end_global = block_start_global + BLOCK_SIZE;
        if (block_start_global >= nbEle)
            continue; // Skip empty padding blocks
        if (block_end_global > nbEle)
            block_end_global = nbEle; // Clip block end

        int max_quant_residual_block = 0;
        unsigned int current_sign_flag_block = 0;

        for (int j_global = block_start_global; j_global < block_end_global; j_global++)
        {
            max_quant_residual_block = max_int(max_quant_residual_block, absQuantResidual[j_global]);
            float prediction = lorenzo_pred(decLocal, j_global, dim_x);
            float residual = oriData[j_global] - prediction;
            float residual_recip = residual * recip_precision;
            int s = residual_recip >= -0.5f ? 0 : 1;
            int curr_quant_residual = (int)(residual_recip + 0.5f) - s;
            int sign_ofs = j_global % BLOCK_SIZE;
            current_sign_flag_block |= (curr_quant_residual < 0) << (BLOCK_SIZE - 1 - sign_ofs);
        }

        // Only ONE thread writes metadata for this block_idx - no fighting for memory address
        int temp_fixed_rate = max_quant_residual_block == 0 ? 0 : (sizeof(int) * 8 - __builtin_clz((unsigned int)max_quant_residual_block));
        signFlag[block_idx] = current_sign_flag_block;
        fixedRate[block_idx] = temp_fixed_rate;
        cmpData[block_idx] = (unsigned char)temp_fixed_rate; // Write rate byte header part
    } // End Parallel Pass 2

    // --- Calculate Block Data Offsets (Serial Prefix Sum) ---
    // Required because Pass 3 is parallelized by block
    size_t *block_data_offsets = malloc(total_blocks * sizeof(size_t));
    if (!block_data_offsets)
    {
        perror("malloc failed for offsets");
        free(decLocal);
        *cmpSize = 0;
        return;
    }
    size_t current_offset = total_blocks; // Data starts after rate bytes header
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        block_data_offsets[block_idx] = current_offset;
        int rate = fixedRate[block_idx]; // Rate computed in Pass 2
        current_offset += rate ? (4 + (rate * BLOCK_SIZE) / 8) : 0;
    }
    *cmpSize = current_offset; // Final total size calculated serially (not sure how badly this affects performance)

// --- Pass 3: Data Packing (Block Parallel - Block Assignment) ---
#pragma omp parallel for schedule(static) // Distribute BLOCKS among threads
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        int block_start_global = block_idx * BLOCK_SIZE;
        int block_pack_end_global = block_start_global + BLOCK_SIZE;                                // Use full block for packing structure
        int block_eff_end_global = (block_pack_end_global > nbEle) ? nbEle : block_pack_end_global; // Limit reads
        if (block_start_global >= nbEle)
            continue;

        int temp_fixed_rate = fixedRate[block_idx];
        unsigned int current_sign_flag = signFlag[block_idx];
        size_t cmp_byte_ofs = block_data_offsets[block_idx]; // Get pre-calculated offset

        // Pack data if rate > 0
        if (temp_fixed_rate > 0)
        {
            // Write sign information (4 bytes)
            cmpData[cmp_byte_ofs++] = (unsigned char)(current_sign_flag >> 24);
            cmpData[cmp_byte_ofs++] = (unsigned char)(current_sign_flag >> 16);
            cmpData[cmp_byte_ofs++] = (unsigned char)(current_sign_flag >> 8);
            cmpData[cmp_byte_ofs++] = (unsigned char)(current_sign_flag);

            // Pack absQuantResidual data (rate * 4 bytes)
            unsigned char tmp_chars[4];
            for (int j = 0; j < temp_fixed_rate; j++)
            { // Loop bit planes
                tmp_chars[0] = tmp_chars[1] = tmp_chars[2] = tmp_chars[3] = 0;
                int bit_pos = j;
                for (int k_idx = 0; k_idx < 8; ++k_idx)
                { // Loop elements within segments
                    if (block_start_global + k_idx < block_eff_end_global)
                        tmp_chars[0] |= ((absQuantResidual[block_start_global + k_idx] >> bit_pos) & 1) << (7 - k_idx);
                    if (block_start_global + 8 + k_idx < block_eff_end_global)
                        tmp_chars[1] |= ((absQuantResidual[block_start_global + 8 + k_idx] >> bit_pos) & 1) << (7 - k_idx);
                    if (block_start_global + 16 + k_idx < block_eff_end_global)
                        tmp_chars[2] |= ((absQuantResidual[block_start_global + 16 + k_idx] >> bit_pos) & 1) << (7 - k_idx);
                    if (block_start_global + 24 + k_idx < block_eff_end_global)
                        tmp_chars[3] |= ((absQuantResidual[block_start_global + 24 + k_idx] >> bit_pos) & 1) << (7 - k_idx);
                }
                // Write the 4 bytes for this bit plane
                cmpData[cmp_byte_ofs++] = tmp_chars[0];
                cmpData[cmp_byte_ofs++] = tmp_chars[1];
                cmpData[cmp_byte_ofs++] = tmp_chars[2];
                cmpData[cmp_byte_ofs++] = tmp_chars[3];
            }
        }
    } // End Parallel Pass 3

    free(block_data_offsets);
    free(decLocal);
}

// --- Decompression Kernel with Wavefront & Block Parallel Passes ---
void hawkZip_decompress_kernel(float *decData, unsigned char *cmpData, int *absQuantResidual, int *fixedRate,
                               /* unsigned int *threadOfs, // No longer needed */
                               size_t nbEle, float errorBound)
{
    if (nbEle == 0)
        return;

    // --- Setup ---
    const size_t dim_x = DIM_X;
    const size_t dim_y = (nbEle + dim_x - 1) / dim_x;
    const int total_blocks = (nbEle + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *signFlag = (unsigned int *)calloc(total_blocks, sizeof(unsigned int));
    if (!signFlag)
    {
        perror("calloc failed for signFlag");
        memset(decData, 0, nbEle * sizeof(float));
        return;
    }

    omp_set_num_threads(NUM_THREADS); // Set threads for subsequent parallel regions

// --- Pass 1: Read Rates (Block Parallel - Block Assignment) ---
#pragma omp parallel for schedule(static) // Distribute BLOCKS among threads
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        // Read rate byte - Safe read
        int temp_fixed_rate = (int)cmpData[block_idx];
        fixedRate[block_idx] = temp_fixed_rate; // Write to unique index - Safe
    } // End Parallel Pass 1

    // --- Calculate Block Data Offsets (Serial Prefix Sum) ---
    size_t *block_data_offsets = malloc(total_blocks * sizeof(size_t));
    if (!block_data_offsets)
    {
        perror("malloc failed for offsets");
        free(signFlag);
        memset(decData, 0, nbEle * sizeof(float));
        return;
    }
    size_t current_offset = total_blocks; // Data starts after rate bytes header
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        block_data_offsets[block_idx] = current_offset;
        int rate = fixedRate[block_idx]; // Rate computed in Pass 1
        current_offset += rate ? (4 + (rate * BLOCK_SIZE) / 8) : 0;
    }

// --- Pass 2: Unpack Signs and Residuals (Block Parallel - Block Assignment) ---
#pragma omp parallel for schedule(static) // Distribute BLOCKS among threads
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        int block_start_global = block_idx * BLOCK_SIZE;
        int block_eff_end_global = block_start_global + BLOCK_SIZE; // Use full block structure
        if (block_eff_end_global > nbEle)
            block_eff_end_global = nbEle; // Limit reads/writes
        if (block_start_global >= nbEle)
            continue;

        int temp_fixed_rate = fixedRate[block_idx];
        size_t cmp_byte_ofs = block_data_offsets[block_idx]; // Get pre-calculated offset

        // Unpack Block Data if rate > 0
        if (temp_fixed_rate > 0)
        {
            // Unpack sign information - Write to unique index is safe
            signFlag[block_idx] = (0xff000000 & ((unsigned int)cmpData[cmp_byte_ofs++] << 24)) |
                                  (0x00ff0000 & ((unsigned int)cmpData[cmp_byte_ofs++] << 16)) |
                                  (0x0000ff00 & ((unsigned int)cmpData[cmp_byte_ofs++] << 8)) |
                                  (0x000000ff & (unsigned int)cmpData[cmp_byte_ofs++]);

            // Zero absQuantResidual for the current block elements before unpacking
            for (int j_global = block_start_global; j_global < block_eff_end_global; ++j_global)
            {
                absQuantResidual[j_global] = 0;
            }

            // Unpack absQuantResidual data
            unsigned char tmp_chars[4];
            for (int j = 0; j < temp_fixed_rate; j++)
            { // Loop bit planes
                tmp_chars[0] = cmpData[cmp_byte_ofs++];
                tmp_chars[1] = cmpData[cmp_byte_ofs++];
                tmp_chars[2] = cmpData[cmp_byte_ofs++];
                tmp_chars[3] = cmpData[cmp_byte_ofs++];
                int bit_pos = j;
                for (int k_idx = 0; k_idx < 8; ++k_idx)
                { // Loop elements within segments
                    if (block_start_global + k_idx < block_eff_end_global)
                        absQuantResidual[block_start_global + k_idx] |= ((tmp_chars[0] >> (7 - k_idx)) & 1) << bit_pos;
                    if (block_start_global + 8 + k_idx < block_eff_end_global)
                        absQuantResidual[block_start_global + 8 + k_idx] |= ((tmp_chars[1] >> (7 - k_idx)) & 1) << bit_pos;
                    if (block_start_global + 16 + k_idx < block_eff_end_global)
                        absQuantResidual[block_start_global + 16 + k_idx] |= ((tmp_chars[2] >> (7 - k_idx)) & 1) << bit_pos;
                    if (block_start_global + 24 + k_idx < block_eff_end_global)
                        absQuantResidual[block_start_global + 24 + k_idx] |= ((tmp_chars[3] >> (7 - k_idx)) & 1) << bit_pos;
                }
            }
        }
        else
        {
            // If rate is 0, ensure residuals for the block elements are 0
            for (int j_global = block_start_global; j_global < block_eff_end_global; ++j_global)
            {
                absQuantResidual[j_global] = 0;
            }
        }
    } // End Parallel Pass 2

    free(block_data_offsets); // Free offsets calculated for Pass 2

    // --- Pass 3: Wavefront Reconstruction ---
    // This pass uses the results from Pass 1 (fixedRate) and Pass 2 (signFlag, absQuantResidual)
    // It writes the final results to decData
    for (size_t d = 0; d < dim_y + dim_x - 1; ++d)
    { // Loop diagonals
        size_t r_start = (d >= dim_x) ? (d - dim_x + 1) : 0;
        size_t r_end = (d < dim_y) ? d : (dim_y - 1);
#pragma omp parallel for schedule(static) // Parallelize WITHIN diagonal
        for (size_t r = r_start; r <= r_end; ++r)
        { // Canonical loop form
            size_t c = d - r;
            if (c >= dim_x)
                continue;
            size_t idx = r * dim_x + c;
            if (idx >= nbEle)
                continue;

            float prediction = lorenzo_pred(decData, idx, dim_x);
            int block_idx = idx / BLOCK_SIZE;
            int sign_ofs = idx % BLOCK_SIZE;
            int temp_fixed_rate = fixedRate[block_idx];
            unsigned int current_sign_flag = signFlag[block_idx];
            int current_abs_quant = absQuantResidual[idx];
            int currQuantResidual = (temp_fixed_rate == 0) ? 0 : ((current_sign_flag & (1U << (BLOCK_SIZE - 1 - sign_ofs))) ? -current_abs_quant : current_abs_quant);
            float decompressed_residual = (float)currQuantResidual * errorBound * 2.0f;
            decData[idx] = prediction + decompressed_residual;
        }
    } // End Wavefront Pass 3

    free(signFlag);
}