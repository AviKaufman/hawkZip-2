# hawkZip: Lossy Compression for Floating-Point Data

## Introduction
HawkZip is a CPU error-bounded lossy compressor designed for floating-point data, developed as part of UIowa: CS:4700. It provides efficient data compression while maintaining accuracy under user-specified error bounds. The goal of this final project is to use any materials from this course to improve **throughput** and **compression ratio** of this compressor.

## Install/Compile hawkZip
System Requirements
- A Linux machine with GCC compiler installed.
- OpenMP support for parallel processing. You can install OpenMP with following commands:
```shell
sudo apt-get install libgomp1  # Debian/Ubuntu
sudo yum install libgomp       # CentOS/RHEL
```

To compile HawkZip, use the following command:
```shell
gcc hawkZip_main.c -O0 -o hawkZip -lm -fopenmp
```
Consider LLVM may be tricky to configure on local machine, we use ```gcc``` compiler for development. Final evaluation of hawkZip wlil also be compiled with ```-O0``` optimization.

## Example of hawkZip Execution
```shell
Usage:
  ./hawkZip -i [srcFilePath] -e [errorBound] [-x cmpFilePath] [-o decFilePath]

Options:
  -i [srcFilePath]  Required. Input file path of original data
  -e [errorBound]   Required. Relative error bound
  -x [cmpFilePath]  Optional. Output file path of compressed data
  -o [decFilePath]  Optional. Output file path of decompressed data

Example:
  ./hawkZip -i original_file.f32 -e 1e-2
  ./hawkZip -i original_file.f32 -e 1e-4 -x comprssed.bin
  ./hawkZip -i original_file.f32 -e 0.01 -o decompressed.bin
  ./hawkZip -i original_file.f32 -e 0.35 -x compressed.bin -o decompressed.bin
```

## Compression Algorithm Explanation
hawkZip is a parallel error-bounded lossy compressor that is implemented using the idea of *Shared Address Space* via OpenMP parallel programming interface.

Given an original floating-point data, within any dimensions, computer system store them as 1D array in memory system.
After dividing data into a set of data blocks (each block contains 32 consecutive floating point numbers), hawkZip generate compressed data via four major steps.

1. Quantization: convert each floating-point number into a corresponding quantized integer based on the error bound. This guarantees error bound, and all following steps are lossless compression for integers.
2. Fixed-length encoding: For integers inside a data block, based on the greatest absolute integer, preserve a fixed-number of bit for all integers (i.e. removing all unncessary zero bits).
3. Block concatenation: concatenating all compressed data block together into a single, unifed byte array.

Decompression works in a reverse manner.
Note that the parallelisms are operated at data block granularity, and more detailed implementations can be checked with code (as well as comments) for functions ```hawkZip_compress_kernel()``` and ```hawkZip_decompress_kernel()``` in hawkZip_compressor.h.


## Tips and Requirements for Your Implementation
Tips for compression/decompression throughput improvements:
- Vectorization: Optimize read, computation, and write operations using AVX intrinsics to enhance parallelism.
- Thread Optimization: Fine-tune the number of threads, as both increasing and decreasing the thread count may impact performance.
- Efficient Parallel Strategy: Improve problem decomposition, task assignment, orchestration, and hardware mapping to maximize efficiency.
- Compiler Optimizations: Leverage additional compiler-level optimizations to enhance performance.

Tips for improving compression ratio:
- Delta Encoding: Store differences between consecutive values instead of raw data to improve compressibility.
- Advanced Delta Encoding: Extend delta encoding by incorporating spatial locality for better data reduction.
- Block Size Optimization: The current block size is hardcoded to 32; adjusting this parameter may lead to better compression ratios.

To evaluate this compressor, please use CESM-ATM, the standard climate simulation dataset from Scientific Data Reduction Benchmark Suites (SDRBench: [Link](https://sdrbench.github.io/)). The dataset download commands can be shown as below:
```shell
# Download dataset
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz

# Extract all fields in this dataset (79 fields in total)
tar -xvf SDRBENCH-CESM-ATM-1800x3600.tar.gz

# Compress a sample field using hawkZip with error bound 1e-4
./hawkZip -i ./1800x3600/TAUX_1_1800_3600.f32 -e 1e-4
```

Note that only **hawkZip_compressor.h** and **hawkZip_entry.h** can be modified.
Besides the tips mentioned above, you may use external research papers and resources but must justify and cite your sources properly.
Below are some research papers for CPU lossy compressors that may provide insights and inspiration for improving this compressor.
- [IPDPS'16] Fast Error-bounded Lossy HPC Data Compression with SZ
- [IPDPS'17] Significantly Improving Lossy Compression for Scientific Data Sets Based on Multidimensional Prediction and Error-Controlled Quantization
- [ICDE'21] Optimizing Error-Bounded Lossy Compression for Scientic Data by Dynamic Spline Interpolation
- [HPDC'22] Ultra-fast Error-bounded Lossy Compression for Scientific Dataset
- [IPDPS'25] Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds

## Grading Principles
Your final project will be evaluated based on the following criteria:

1. Basic Coding Score (30%)
You are required to submit a functionally correct and fully operational implementation of the lossy compressor.
The code should run without errors and produce meaningful compression results.
A well-structured, maintainable, and readable codebase is encouraged.
2. Performance Ranking (20%)
All submitted implementations will be benchmarked and ranked based on two key performance metrics:
Compression Ratio – How much the data size is reduced while passing the error check (```line 108~121``` in ```hawZip_main.c```).
Throughput – The speed of compression and decompression (GB/) while passing the aforementioned error check.
The ranking will be determined based on a fair and consistent evaluation framework, and scores will be assigned accordingly.
3. Project Report (30%)
Your report should provide a comprehensive evaluation of your compressor, including:
Introduction – A brief overview of the approach used in your implementation.
Methodology – A detailed description of the techniques and optimizations applied.
Evaluation – An analysis of your compressor’s throughput and compression ratio with supporting data.
References – Properly cite any external sources, papers, or tools used.
Optional but encouraged: Additional quality assessment metrics such as visualization, PSNR, and SSIM for evaluating data quality (quality can be evaluated using QCAT [link](https://github.com/szcompressor/qcat)).
4. Group Presentation (20%)
Each group will present their implementation, results, and key takeaways in class.
The presentation should clearly articulate the design choices, optimization strategies, and experimental findings.
Effective communication, clarity of explanation, and well-structured slides will contribute to the score.

**Academic Integrity & Collaboration Policy**:
Cross-group collaboration is strictly prohibited. Each team must develop its own unique solution.
Use of Large Language Models (LLMs) is strictly prohibited.
Any form of plagiarism or unauthorized code-sharing will result in penalties, as per academic integrity policies.
We will check AI generated contents using our tool to detect plagiarism.
If you have any questions regarding the grading criteria or project expectations, feel free to reach out.

## Submission Guidelines
1. Submission Format & Naming
    Submit a single .zip file named:
    ```
    hawkZip_<group_name>.zip
    Example: hawkZip_TeamAlpha.zip
    ```
    Inside the .zip, include:
    (1) ```src/``` folder – hawkZip code. (2) report.pdf – Must include group member names on the first page. (3) README.md – Instructions on compiling and running your code in case you use other external dependencies.
2. Submission & Deadlines
Upload to ICON before April 25 (11:59 PM CST).
Presentations start on April 30.
3. Notes
Ensure your code compiles and runs correctly before submission.
No late resubmissions will be accepted.


## Contact & Support
Yafan Huang: *yafan-huang@uiowa.edu*