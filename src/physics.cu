
extern "C" {
#include "physics.h"
}

#include "solve.cu"

#include <stdio.h>

#include <curand.h>


#define RESERVE 2000


static size_t reserved = 0;

static float* d_balls = NULL;


void reserve(size_t size)
{
    reserved = size;

    if (d_balls != NULL)
    {
        cudaFree(d_balls);
    }

    if (cudaMalloc(&d_balls, size * SIZE * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "(CUDA) ERROR: allocation failed\n");
    }
}


extern "C"
int physics_init()
{
    const int device = 0;

    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "(CUDA) ERROR: cannot set CUDA device: %s\n",
                            cudaGetErrorString(err));
            return 1;
        }
    }

    {
        void* ptr = NULL;
        cudaError_t err = cudaMalloc(&ptr, 1024);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "(CUDA) ERROR: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("(CUDA) Using device %d: '%s'\n", device, deviceProp.name);

    reserve(RESERVE);

    return 0;
}


extern "C"
void physics_quit()
{
    cudaFree(d_balls);
}


extern "C"
void solve_circles(float* balls, int count,
                   float gravity,
                   float minx, float maxx,
                   float miny, float maxy)
{
    while (count > reserved)
    {
        reserve(2 * reserved);
    }

    cudaMemcpy(d_balls, balls, count * SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    solve_gpu(d_balls, count, gravity, minx, maxx, miny, maxy);

    cudaMemcpy(balls, d_balls, count * SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);
}
