#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab3IO.h"
#include "timer.h"
#include <omp.h>

int main(int argc, char* argv[]) {
    int size;
    double** matrix;
    double start, end;
    double temp;
    int i, j, k;
    int num_threads;

    // Check if the number of threads is provided
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
        return 1;
    }

    num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        fprintf(stderr, "Number of threads must be positive\n");
        return 1;
    }

    // Set the number of threads
    omp_set_num_threads(num_threads);

    // Load input data
    if (Lab3LoadInput(&matrix, &size)) {
        fprintf(stderr, "Error loading matrix\n");
        return 1;
    }

    GET_TIME(start);

    // Gauss-Jordan Elimination with Partial Pivoting
    for (k = 0; k < size - 1; k++) {
        // Partial Pivoting
        int max = k;
        #pragma omp parallel for private(i) shared(matrix, size, k) schedule(static)
        for (i = k + 1; i < size; i++) {
            if (fabs(matrix[i][k]) > fabs(matrix[max][k])) {
                #pragma omp critical
                max = i;
            }
        }

        // Swap rows
        if (max != k) {
            double* tempPtr = matrix[k];
            matrix[k] = matrix[max];
            matrix[max] = tempPtr;
        }

        // printf("After swapping:\n");
        // for (i = 0; i < size; ++i) {
        //     for (j = 0; j < size + 1; ++j) {
        //         printf("%f ", matrix[i][j]);
        //     }
        //     printf("\n");
        // }

        // Elimination
        #   pragma omp parallel for private(i, j, temp) shared(matrix, size, k) schedule(static)
        for (i = k + 1; i < size; ++i) {
            temp = matrix[i][k] / matrix[k][k];
            for (j = k; j <= size; ++j) {
                matrix[i][j] -= temp * matrix[k][j];
            }
        }

        // printf("After elimination:\n");
        // for (i = 0; i < size; ++i) {
        //     for (j = 0; j < size + 1; ++j) {
        //         printf("%f ", matrix[i][j]);
        //     }
        //     printf("\n");
        // }

        // printf("\n\n");

    }

    // printf("FINAL FORM:\n");
    // for (i = 0; i < size; ++i) {
    //     for (j = 0; j < size + 1; ++j) {
    //         printf("%f ", matrix[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // // Normalize the diagonal
    // #pragma omp parallel for private(i, j) shared(matrix, size) schedule(auto)
    // for (i = 0; i < size; ++i) {
    //     temp = matrix[i][i];
    //     for (j = i; j < size + 1; ++j) {
    //         matrix[i][j] /= temp;
    //     }
    // }
    #   pragma omp parallel for private(i, k) shared(matrix, size) schedule(static)
    for (k = size-1; k >= 1; k--) {
        for (i = 0; i < k; i++) {
            matrix[i][size] -= (matrix[i][k]/matrix[k][k])*matrix[k][size];
            matrix[i][k] = 0;
        }
    }

    // printf("RREF FORM:\n");
    // for (i = 0; i < size; ++i) {
    //     for (j = 0; j < size + 1; ++j) {
    //         printf("%f ", matrix[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    GET_TIME(end);

    // Extract solution
    double* solution = (double*) malloc(size * sizeof(double));
    for (i = 0; i < size; ++i) {
        solution[i] = matrix[i][size];
    }

    Lab3SaveOutput(solution, size, end - start);

    for (i = 0; i < size; ++i) {
        free(matrix[i]);
    }
    free(matrix);
    free(solution);

    return 0;
}