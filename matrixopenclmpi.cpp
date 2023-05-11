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

    // Allocate memory on the device
    A = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    B = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    C = (double *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));

    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), NULL, &err);
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), NULL, &err);
    c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), NULL, &err);

    // Initialize matrices A and B
    for (i = 0; i < MATRIX_SIZE; i++)
    {
        for (j = 0; j < MATRIX_SIZE; j++)
        {
            A[i * MATRIX_SIZE + j] = 1.0;
            B[i * MATRIX_SIZE + j] = 1.0;
        }
    }

    // Transfer data to the device
    err = clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, MATRIX_SIZEMATRIX_SIZEsizeof(double), B, 0, NULL, NULL);

    // Set the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    int width = MATRIX_SIZE;
    err = clSetKernelArg(kernel, 3, sizeof(int), &width);

    // Set the global and local work sizes
    global_work_size[0] = MATRIX_SIZE;
    global_work_size[1] = MATRIX_SIZE;
    local_work_size[0] = 32;
    local_work_size[1] = 32;

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // Transfer the results back to the host
    err = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), C, 0, NULL, NULL);

    // Free memory on the device
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);

    // Free memory on the host
    free(A);
    free(B);
    free(C);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}