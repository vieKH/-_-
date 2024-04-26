/*
 *  file name: kernel.cu
 *
 *  kernel.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#define BLOCK_SIZE 32

struct statistical_para {
    double std;
    double mean;
    double confi_inter_left;
    double confi_inter_right;
};

statistical_para count_statistic(int* matrix, int size) {
    statistical_para ret{};
    int sum = 0;
    double sum_sq = 0;

    // Calculate sum of elements
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sum += matrix[i * size + j];
        }
    }
    ret.mean = static_cast<double>(sum) / (size * size);

    // Calculate sum of squared differences from mean
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sum_sq += (static_cast<double>(matrix[i * size + j]) - ret.mean) *
                (static_cast<double>(matrix[i * size + j]) - ret.mean);
        }
    }
    ret.std = sqrt(sum_sq);

    // Calculate confidence interval
    ret.confi_inter_left = ret.mean - 1.96 * (ret.std / sqrt(size));
    ret.confi_inter_right = ret.mean + 1.96 * (ret.std / sqrt(size));

    return ret;
}

__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_square_matrix_mult(int* d_a, int* d_b, int* d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n * n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


void cpu_matrix_mult(int* h_a, int* h_b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void write_matrix_to_file(int* matrix, int size, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(file, "%d ", matrix[i * size + j]);
        }
    }

    fclose(file);
}

void algorith_main(int n) {
    srand(3333);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int* h_a, * h_b, * h_c, * h_cc;
    cudaMallocHost((void**)&h_a, sizeof(int) * n * n);
    cudaMallocHost((void**)&h_b, sizeof(int) * n * n);
    cudaMallocHost((void**)&h_c, sizeof(int) * n * n);
    cudaMallocHost((void**)&h_cc, sizeof(int) * n * n);

    // random initialize matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }
    write_matrix_to_file(h_a, n, "matrixA.txt");

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_b[i * n + j] = rand() % 1024;
        }
    }
    write_matrix_to_file(h_b, n, "matrixB.txt");

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    printf("Block size is: %d\n", BLOCK_SIZE);

    for (int size = 100; size <= n; size += 100)
    {
        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start to count execution time of GPU version

        cudaEventRecord(start, 0);
        clock_t start_gpu = clock();

        // Allocate memory space on the device 
        int* d_a, * d_b, * d_c;
        cudaMalloc((void**)&d_a, sizeof(int) * size * size);
        cudaMalloc((void**)&d_b, sizeof(int) * size * size);
        cudaMalloc((void**)&d_c, sizeof(int) * size * size);

        // copy matrix A and B from host to device memory
        cudaMemcpy(d_a, h_a, sizeof(int) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * size * size, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Launch kernel 


        gpu_square_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, size);

        // Transefr results from device to host 
        cudaMemcpy(h_c, d_c, sizeof(int) * size * size, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate

        // compute time elapse on GPU computing

        gpu_elapsed_time_ms = clock() - start_gpu;

        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f s\n", size, size, size, size, (gpu_elapsed_time_ms / CLOCKS_PER_SEC) * pow(10.0, 6) / 1000 / 1000);



        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        // printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f s - using CudaEvent\n\n", n, n, n, n, gpu_elapsed_time_ms / 1000);


        clock_t start_cpu = clock();

        cpu_matrix_mult(h_a, h_b, h_cc, size, size, size);

        cpu_elapsed_time_ms = clock() - start_cpu;

        printf("Time elapsed on matrix multiplication on CPU: %f s.\n\n", (cpu_elapsed_time_ms / CLOCKS_PER_SEC) * pow(10.0, 6) / 1000 / 1000);
        
        if (size == n)
        {
            write_matrix_to_file(h_c, n, "matrix_result.txt");

            statistical_para st;
            st = count_statistic(h_a, size);

            printf("\nStatistical parameters for matrix A\n");
            printf("Mean of matrix: %.2f\n", st.mean);
            printf("Standard deviation of matrix: %.2f\n", st.std);
            printf("Confidence interval with 95%% confidence: (%.2f, %.2f)\n", st.confi_inter_left, st.confi_inter_right);

            st = count_statistic(h_b, size);

            printf("\nStatistical parameters for matrix B\n");
            printf("Mean of matrix: %.2f\n", st.mean);
            printf("Standard deviation of matrix: %.2f\n", st.std);
            printf("Confidence interval with 95%% confidence: (%.2f, %.2f)\n", st.confi_inter_left, st.confi_inter_right);

        }
    }


}

int main(int argc, char const* argv[])
{
    algorith_main(1000);
    return 0;
}