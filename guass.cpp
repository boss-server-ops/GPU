#include <stdio.h>
#include <stdlib.h>

void ResetMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j < i) {
                matrix[i * N + j] = 0;
            } else if (j == i) {
                matrix[i * N + j] = 1.0;
            } else {
                matrix[i * N + j] = (float)(rand() % 100);
            }
        }
    }
}

__global__ void divideMatrixRow(float* matrix, int row, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        matrix[row * N + i] /= matrix[row * N + row];
    }
}

__global__ void eliminateMatrixRow(float* matrix, int row, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        if (i > row) {
            matrix[i * N + row] = 0;
        }
    }
}

int main() {
    int grid = 256;
    int block = 1024;
    int N = 2000;
    float* matrix;
    int size = N * N * sizeof(float);
    cudaMallocManaged(&matrix, size);
    ResetMatrix(matrix, N);
    cudaDeviceSynchronize();
    cudaError_t ret;

    for (int k = 0; k < N; k++) {
        divideMatrixRow<<<grid, block>>>(matrix, k, N);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("divideMatrixRow failed: %s\n", cudaGetErrorString(ret));
        }

        eliminateMatrixRow<<<grid, block>>>(matrix, k, N);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("eliminateMatrixRow failed: %s\n", cudaGetErrorString(ret));
        }
    }

    cudaFree(matrix);
    return 0;
}
