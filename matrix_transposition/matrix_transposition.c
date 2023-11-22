#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 10000000
#define COLS 50000
#define BLOCK_SIZE 100

void matT(int rows, int cols, double **A, double **result)
{
    double start_time = omp_get_wtime();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result[j][i] = A[i][j];
        }
    }
    double end_time = omp_get_wtime();
    printf("%f;", end_time - start_time);
}

void matBlockT(int rows, int cols, int blockSize, double **A, double **result){
    double start_time = omp_get_wtime();
    for (int i = 0; i < rows; i += blockSize){
        for (int j = 0; j < cols; j += blockSize){
            for (int x = i; x < i + blockSize && x < rows; ++x){
                for (int y = j; y < j + blockSize && y < cols; ++y){
                    result[y][x] = A[x][y];
                }
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("%f;", end_time - start_time);
}

void matTpar(int rows, int cols, double **A, double **result){
    double start_time = omp_get_wtime();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            result[j][i] = A[i][j];
        }
    }
    double end_time = omp_get_wtime();
    printf("%f;", end_time - start_time);
}

void matBlockTpar(int rows, int cols, int blockSize, double **A, double **result){
    double start_time = omp_get_wtime();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += blockSize){
        for (int j = 0; j < cols; j += blockSize){
            for (int x = i; x < i + blockSize; ++x){
                for (int y = j; y < j + blockSize; ++y){
                    result[y][x] = A[x][y];
                }
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("%f;", end_time - start_time);
}

void freeMatrix(double **matrix, int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        free(matrix[i]);
    }
    free(matrix);
}

int main()
{
    double **A;
    double **result;
    int matrix;
    int block;
    for (matrix = 10000; matrix <= 80000; matrix = matrix * 2)
    {
        printf("Matrix size: %d\n", matrix);

        A = (double **)malloc(matrix * sizeof(double *));
        result = (double **)malloc(matrix * sizeof(double *));

        for (int i = 0; i < matrix; ++i)
        {
            A[i] = (double *)malloc(matrix * sizeof(double));
        }

        for (int i = 0; i < matrix; ++i)
        {
            result[i] = (double *)malloc(matrix * sizeof(double));
        }

        // Initialize matrix A with some values (you can use your own initialization logic)
        for (int i = 0; i < matrix; ++i)
        {
            for (int j = 0; j < matrix; ++j)
            {
                A[i][j] = i * matrix + j;
            }
        }

/*
        //  Serial matrix transposition
        matT(matrix, matrix, A, result);

        // Serial matrix block transposition
        for (block = 10; block <= 80; block = block * 2)
        {
            matBlockT(matrix, matrix, block, A, result);
        }
*/

        printf("\n");
        for (int i = 1; i <= 64; i = i * 2)
        {
            omp_set_num_threads(i);

            //  Parallel matrix transposition
            matTpar(matrix, matrix, A, result);

            // Parallel matrix block transposition
            for (block = 10; block <= 80; block = block * 2)
            {
                matBlockTpar(matrix, matrix, block, A, result);
            }
            printf("\n");
        }
        // Free allocated memory
        freeMatrix(A, matrix);
        freeMatrix(result, matrix);
    }

    return 0;
}
