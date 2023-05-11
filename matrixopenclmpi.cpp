#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <mpi.h>

#define MATRIX_SIZE 1024

int main(int argc, char **argv)
{
    int size, rank;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem a_mem, b_mem, c_mem;
    cl_int err;
    size_t global_work_size[2], local_work_size[2];
    int i, j, k;
    double *A, *B, *C;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2) {
        fprintf(stderr, "Error: Need at least 2 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Set up OpenCL
    cl_platform_id platform_id;
    cl_uint num_devices;

    // Get the first OpenCL device available on the system
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the kernel program
    const char *source = "__kernel void matmul(__global double *A, __global double *B, __global double *C, int width) {\n"
                         "int i = get_global_id(0);\n"
                         "int j = get_global_id(1);\n"
                         "double sum = 0.0;\n"
                         "for (int k = 0; k < width; k++) {\n"
                         "   sum += A[i*width+k] * B[k*width+j];\n"
                         "}\n"
                         "C[i*width+j] = sum;\n"
                         "}\n";
    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the kernel
    kernel = clCreateKernel(program, "matmul", &err);

    // Allocate memory on the host
    A = (double *)malloc((MATRIX_SIZE / size) * MATRIX_SIZE * sizeof(double));
    B = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    C = (double *)malloc((MATRIX_SIZE / size) * MATRIX_SIZE * sizeof(double));

    if (rank == 0) {
        // Initialize matrix B
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
               
