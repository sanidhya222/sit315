#include <iostream>
#include <mpi.h>

#define N 1000

int main(int argc, char **argv) {

    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int a[N][N], b[N][N], c[N][N];
    int i, j, k;

    // Initialize matrices
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i * j;
            c[i][j] = 0;
        }
    }

    int chunk_size = N / num_procs;
    int remainder = N % num_procs;

    int *sendcounts = new int[num_procs];
    int *displs = new int[num_procs];

    // Calculate sendcounts and displacements for scatterv
    for (i = 0; i < num_procs; i++) {
        sendcounts[i] = chunk_size * N;
        if (i == 0) {
            sendcounts[i] += remainder * N;
        }
        displs[i] = i * chunk_size * N;
        if (i > 0) {
            displs[i] += remainder * N;
        }
    }

    int *local_a = new int[chunk_size * N];
    int *local_c = new int[chunk_size * N];

    // Scatter matrix 'a' to all processes using scatterv
    MPI_Scatterv(a, sendcounts, displs, MPI_INT, local_a, chunk_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix 'b' to all processes
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local matrix multiplication
    for (i = 0; i < chunk_size; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                local_c[i * N + j] += local_a[i * N + k] * b[k][j];
            }
        }
    }

    // Gather results back to the root process using gather and gather v
    if (rank == 0) {
        int *recvcounts = new int[num_procs];
        for (i = 0; i < num_procs; i++) {
            recvcounts[i] = chunk_size * N;
            if (i == 0) {
                recvcounts[i] += remainder * N;
            }
        }
        MPI_Gatherv(local_c, chunk_size * N, MPI_INT, c, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        delete[] recvcounts;

        // Print result matrix
        std::cout << "Result Matrix:" << std::endl;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                std::cout << c[i][j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        MPI_Gatherv(local_c, chunk_size * N, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }

    delete[] local_a;
    delete[] local_c;
    delete[] sendcounts;
    delete[] displs;

    MPI_Final
