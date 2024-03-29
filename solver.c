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
    for (k = 0; k < size; ++k) {
        // Partial Pivoting
        int max = k;
        for (i = k + 1; i < size; i++) {
            if (fabs(matrix[i][k]) > fabs(matrix[max][k])) {
                max = i;
            }
        }
        if (k != max) {
            // Swap rows
            double* tempPtr = matrix[k];
            matrix[k] = matrix[max];
            matrix[max] = tempPtr;
        }

        // Jordan Elimination
        // #pragma omp parallel for private(i, j, temp) shared(matrix, size, k) schedule(dynamic, 10)
        // for (i = k; i < size; ++i) {
        //     if (i == k) continue;
        //     temp = matrix[i][k] / matrix[k][k];
        //     for (j = k; j < size + 1; ++j) {  // Start from k+1 to skip over zeros
        //         matrix[i][j] -= temp * matrix[k][j];
        //     }
        // }

        // setting dynamic static can schedule
        #pragma omp parallel for private(i, j, temp) shared(matrix, size, k)
        // #pragma omp parallel for private(i, j, temp) shared(matrix, size, k) schedule(dynamic, 10)
        for (i = 0; i < size; ++i) {
            if (i != k) {
                temp = matrix[i][k] / matrix[k][k];
                // if (i == k) continue;
                for (j = k; j < size + 1; ++j) {
                    // if (i == k) continue;
                    matrix[i][j] -= temp * matrix[k][j];
                }
            }
        }
    }

    // Normalize the diagonal
    #pragma omp parallel for private(i, j, temp) shared(matrix, size, k)
    // #pragma omp parallel for private(i, j, temp) shared(matrix, size, k) schedule(dynamic, 13)
    for (i = 0; i < size; ++i) {
        temp = matrix[i][i];
        for (j = i; j < size + 1; ++j) {
            matrix[i][j] /= temp;
        }
    }

    // Before formatting, after join
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