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

    int max = 0;
    // Gauss Elimination
    # pragma omp parallel private(k, i) shared(matrix, size)
    {
        for (k = 0; k < size - 1; k++) {
            // Partial Pivoting
            #  pragma omp single 
            {
                max = k;
            }
            
            # pragma omp for private(i)
            for (i = k + 1; i < size; i++) {
                if (fabs(matrix[i][k]) > fabs(matrix[max][k])) {
                    #pragma omp critical
                    max = i;
                }
            }

            #  pragma omp single 
            {
                // Swap rows
                if (max != k) {
                    double* tempPtr = matrix[k];
                    matrix[k] = matrix[max];
                    matrix[max] = tempPtr;
                }
            }

            // Elimination
            #   pragma omp for private(i, j, temp) schedule(static)
            for (i = k + 1; i < size; ++i) {
                temp = matrix[i][k] / matrix[k][k];
                // #   pragma omp for schedule(static)
                for (j = k; j <= size; ++j) {
                    matrix[i][j] -= temp * matrix[k][j];
                }
            }

        }

        // Jordan Elimination
        #   pragma omp for private(i, k) schedule(static)
        for (k = size-1; k >= 1; k--) {
            // #   pragma omp for schedule(static)
            for (i = 0; i < k; i++) {
                    matrix[i][size] -= (matrix[i][k]/matrix[k][k])*matrix[k][size];
                    matrix[i][k] = 0;
                }
        }

        // Normalize the diagonal
        # pragma omp for private(i, j)
        for (i = 0; i < size; ++i) {
            temp = matrix[i][i];
            for (j = i; j < size + 1; ++j) {
                matrix[i][j] /= temp;
            }
        }
    }

    GET_TIME(end);


    // Extract solution
    double* solution = (double*) malloc(size * sizeof(double));
    for (i = 0; i < size; ++i) {
        solution[i] = matrix[i][size];
    }

    Lab3SaveOutput(solution, size, end - start);
    DestroyVec(solution);
    DestroyMat(matrix, size);

    return 0;
}