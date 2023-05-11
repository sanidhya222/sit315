#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int n = 5000; // Size of the matrix
    int p, rank, num_threads;
    double *A, *B, *C, *local_A, *local_C;
    double start_time, end_time;
    int max_rand_value = 300;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check that number of processes divides the matrix size
    if (n % p != 0)
    {
        if (rank == 0)
        {
            printf("Error: Number of processes %d does not divide matrix size %d\n", p, n);
        }
        MPI_Finalize();
        exit(1);
    }

    // Allocate memory for matrices
    A = (double *)malloc(n * n * sizeof(double));
    B = (double *)malloc(n * n * sizeof(double));
    C = (double *)malloc(n * n * sizeof(double));
    local_A = (double *)malloc(n * n / p * sizeof(double));
    local_C = (double *)malloc(n * n / p * sizeof(double));

    // Read matrices from file or accept user input
    // Alternatively, you can keep the code that generates random matrices as before
    // ...

    // Distribute matrix A among processes
    MPI_Scatter(A, n * n / p, MPI_DOUBLE, local_A, n * n / p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication using OpenMP
    start_time = MPI_Wtime();
#pragma omp parallel private(num_threads)
    {
        num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int num_elements = n / p;
        int row_offset = (rank * num_elements) + (tid * (num_elements / num_threads));

        // Use static scheduling with chunk size of 1
        #pragma omp for schedule(static, 1)
        for (int i = row_offset; i < row_offset + (num_elements / num_threads); i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0
