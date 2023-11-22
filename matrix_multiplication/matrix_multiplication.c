#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function to allocate memory for a matrix
float **allocateMatrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; ++i)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrix;
}

// Function to free memory allocated for a matrix
void freeMatrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to validate matrix compatibility for multiplication
int validateMatrixDimensions(int cols_A, int rows_B)
{
    if (cols_A != rows_B)
    {
        printf("Error: Incompatible matrix dimensions for multiplication.\n");
        return 0; // Return 0 to indicate failure
    }
    return 1; // Return 1 to indicate success
}

// Serial matrix multiplication
void matMul(float **A, float **B, float **C, int rows_A, int cols_A, int cols_B)
{
    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            C[i][j] = 0.0;
            for (int k = 0; k < cols_A; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP
void matMulPar(float **A, float **B, float **C, int rows_A, int cols_A, int cols_B)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
            for (int k = 0; k < cols_A; ++k)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main()
{
    // Define matrix dimensions
    int matrixSize;
    int threads;
    for (matrixSize = 1024; matrixSize <= 8192; matrixSize *= 2)
    {

        // Allocate memory for matrices
        float **A = allocateMatrix(matrixSize, matrixSize);
        float **B = allocateMatrix(matrixSize, matrixSize);
        float **C = allocateMatrix(matrixSize, matrixSize);

        // For simplicity, we'll use sequential values here
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                A[i][j] = i * matrixSize + j + 1;
            }
        }

        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                B[i][j] = i * matrixSize + j + 1;
            }
        }

        // Validate matrix dimensions
        if (!validateMatrixDimensions(matrixSize, matrixSize))
        {
            // Handle error and exit
            return 1;
        }

        // Perform serial matrix multiplication and measure time
        double start_serial = omp_get_wtime();
        matMul(A, B, C, matrixSize, matrixSize, matrixSize);
        double end_serial = omp_get_wtime();
        double elapsed_serial = end_serial - start_serial;
        printf("\nSerial Execution Time: %f seconds\n", elapsed_serial);

        for (threads = 2; threads <= 64; threads *= 2)
        {
            omp_set_num_threads(threads);
            // Perform parallel matrix multiplication and measure time
            double start_parallel = omp_get_wtime();
            matMulPar(A, B, C, matrixSize, matrixSize, matrixSize);
            double end_parallel = omp_get_wtime();
            double elapsed_parallel = end_parallel - start_parallel;
            printf("Parallel Execution Time: %f seconds\n", elapsed_parallel);
        }

        // Free allocated memory
        freeMatrix(A, matrixSize);
        freeMatrix(B, matrixSize);
        freeMatrix(C, matrixSize);
    }
    return 0;
}