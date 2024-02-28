#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Lab3IO.h"
#include "timer.h"
#include <omp.h>

int main(void) {
    int size;
    double** matrix;
    double start, end;
    double temp;
    int i, j, k;
    
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
            double* temp = matrix[k];
            matrix[k] = matrix[max];
            matrix[max] = temp;
        }

        // Jordan Elimination
        #pragma omp parallel for private(i, j, temp) shared(matrix, size, k)
        for (i = 0; i < size; ++i) {
            if (i != k) {
                temp = matrix[i][k] / matrix[k][k];
                for (j = k; j < size + 1; ++j) {
                    if (i == k) continue;
                    matrix[i][j] -= temp * matrix[k][j];
                }
            }
        }
    }

    // Normalize the diagonal
    #pragma omp parallel for private(i, j)
    for (i = 0; i < size; ++i) {
        temp = matrix[i][i];
        for (j = i; j < size + 1; ++j) {
            matrix[i][j] /= temp;
        }
    }

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
