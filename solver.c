#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "Lab3IO.h"
#include "timer.h"  // Make sure to include this header for GET_TIME

int main(void) {
    // At the beginning of your main function or where the computation starts
    double start, end, elapsed_time;
    GET_TIME(start);
    double** matrix;
    int size;
    int i, j, k;
    double temp;

    // Load the input data
    if (Lab3LoadInput(&matrix, &size) != 0) {
        fprintf(stderr, "Error loading input!\n");
        return 1;
    }

    // Gauss-Jordan Elimination
    for (k = 0; k < size; k++) {
        #pragma omp parallel for private(j, temp)
        for (i = k + 1; i < size; i++) {
            temp = matrix[i][k] / matrix[k][k];
            for (j = k; j < size + 1; j++) {
                matrix[i][j] -= temp * matrix[k][j];
            }
        }
    }

    // Backward substitution
    for (k = size - 1; k >= 0; k--) {
        matrix[k][size] /= matrix[k][k];
        matrix[k][k] = 1;
        #pragma omp parallel for private(i, j)
        for (i = k - 1; i >= 0; i--) {
            matrix[i][size] -= matrix[i][k] * matrix[k][size];
            matrix[i][k] = 0;
        }
    }

    // Extract the solution
    double* solution = (double*) malloc(sizeof(double) * size);
    for (i = 0; i < size; i++) {
        solution[i] = matrix[i][size];
    }

    // After the computation is finished
    GET_TIME(end);
    elapsed_time = end - start;

    // Save the solution and clean up
    Lab3SaveOutput(solution, size, elapsed_time);
    for (i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(solution);

    return 0;
}
