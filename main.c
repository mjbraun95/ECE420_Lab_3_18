/**
 * @file main.c
 * @author Patricia Zafra, Charles Ancheta
 * @brief Main executable for Lab 3
 * @version 0.1
 * @date 2022-03-10
 *
 * Computes the solution for a system of equations
 * using Gaussian and Jordan elimination
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "Lab3IO.h"
// #include "solver.h"
#include "timer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void usage(const char *prog_name) {
    fprintf(stderr, "USAGE: %s <NUM_THREADS>\n", prog_name);
    exit(1);
}

double **A, *X;
/* My compiler complains about redefining the built-in function index as a non-function so we had to
 * rename this */
int *index_vec, size, thread_count;

/* WARNING: the code that follows will make you cry;
 *          a safety pig is provided below for your benefit
 *                           _
 *   _._ _..._ .-',     _.._(`))
 *  '-. `     '  /-._.-'    ',/
 *     )         \            '.
 *    / _    _    |             \
 *   |  a    a    /              |
 *   \   .-.                     ;
 *    '-('' ).-'       ,'       ;
 *       '-;           |      .'
 *          \           \    /
 *          | 7  .__  _.-\   \
 *          | |  |  ``/  /`  /
 *         /,_|  |   /,_/   /
 *            /,_/      '`-'
 */

/**
 * @brief Find the row that has the maximum absolute value of the element in the kth column
 *
 * @param k the row k in matrix A
 * @return index of the row with maximum
 */
int pivot(int k) {
    int col = k;
    int max = k;

    for (k = 0; k < size; k++) {
        if (fabs(A[index_vec[k]][col]) > A[index_vec[max]][col]) max = k;
    }
    return max;
}

/**
 * @brief Swap two indicies
 *
 * @param index_1
 * @param index_2
 */
void swap(int index_1, int index_2) {
    int temp = index_1;
    index_1 = index_2;
    index_2 = temp;
}

/* Gaussian elimination */
void gaussian() {
    double temp;
    int i, j, k;
    int max = 0;

    /* clang-format off */
    for (k = 0; k < size - 1; k++) {
        /* Pivoting */
        /* Use single thread when finding max and swapping */
        #pragma omp single
        {
            max = pivot(k);
            swap(index_vec[k], index_vec[max]);
        }

        /* Elimination */
        /* Parallelize elimination steps */
        #pragma omp for schedule(static)
        for (i = k + 1; i < size; i++) {
            temp = A[index_vec[i]][k] / A[index_vec[k]][k];
            for (j = k; j < size + 1; j++)
                A[index_vec[i]][j] -= A[index_vec[k]][j] * temp;
        }
    }
    /* clang-format on */
}

/* Jordan elimination */
void jordan() {
    int i, k;

    /* clang-format off */
    for (k = size - 1; k > 0; k--) {
        #pragma omp for schedule(static)
        for (i = k - 1; i >= 0; i--) {
            A[index_vec[i]][size] -=
                A[index_vec[i]][k] / A[index_vec[k]][k] * A[index_vec[k]][size];
            A[index_vec[i]][k] = 0;
        }
    }
    /* clang-format on */
}

double solve() {
    int i;
    double start, end;

    /* Initialize vectors */
    X = CreateVec(size);
    index_vec = malloc(size * sizeof(*index_vec));
    for (i = 0; i < size; ++i)
        index_vec[i] = i;

    /* Start timer */
    GET_TIME(start);

    /* Early return for basic case where size == 1 */
    if (size == 1) {
        X[0] = A[0][1] / A[0][0];
        GET_TIME(end);
        return end - start;
        /* clang-format off */
    }

    #pragma omp parallel num_threads(thread_count) shared(A, index_vec)
    {
        /* Parallelize the algorithms using the same parallel team */
        gaussian();
        jordan();
        /* Compute solution vector */
        #pragma omp for
        for (i = 0; i < size; ++i)
            X[i] = A[index_vec[i]][size] / A[index_vec[i]][i];
    }
    /* clang-format on */
    GET_TIME(end);
    return end - start;
}

int main(int argc, const char **argv) {
    double time;

    if (argc != 2) usage(argv[0]);
    thread_count = atoi(argv[1]);
    if (!thread_count) usage(argv[0]);

    Lab3LoadInput(&A, &size);
    time = solve();
    printf("Time taken: %e seconds\n", time);
    Lab3SaveOutput(X, size, time);
    DestroyVec(X);
    DestroyMat(A, size);

    return 0;
}