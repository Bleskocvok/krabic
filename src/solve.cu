
#include "physics.h"

#define BLOCK 128



__device__
float clamp(float val, float l, float h)
{
    if (val < l)
        return l;
    if (val > h)
        return h;
    return val;
}


__device__
void apply_forces(float* ball, const float gravity)
{
    float dx = ball[X] - ball[PREV_X];
    float dy = ball[Y] - ball[PREV_Y];

    ball[PREV_X] = ball[X];
    ball[PREV_Y] = ball[Y];

    ball[X] += dx;
    ball[Y] += dy;

    ball[Y] += gravity * 0.003;

    float r = ball[RADIUS];
    ball[X] = clamp(ball[X], 0 + r, 600 - r);
    ball[Y] = clamp(ball[Y], 0 + r, 600 - r);
}


__device__
void collision(const float* __restrict__ a,
               const float* __restrict__ b,
               float* result)
{
    float dx = a[X] - b[X];
    float dy = a[Y] - b[Y];

    float d = a[RADIUS] + b[RADIUS];

    if (dx * dx + dy * dy >= d * d)
        return;

    float dist = sqrt(dx * dx + dy * dy);

    float diff_x = dist == 0 ? d / 4 : (dist - d) / dist;
    float diff_y = dist == 0 ? d / 4 : (dist - d) / dist;

    result[0] -= dx * diff_x * 0.5;
    result[1] -= dy * diff_y * 0.5;
}


__device__
void solve_collisions(float* balls, int count,
                      float minx, float maxx,
                      float miny, float maxy)
{
    int ti = blockIdx.x * BLOCK + threadIdx.x;
    float* ball = balls + ti * SIZE;

    float coords[2] = { 0 };

    for (int i = 0; i < count; i++)
    {
        if (ti != i)
        {
            collision(ball, balls + i * SIZE, coords);
        }
    }

    __syncthreads();

    // float r = ball[RADIUS];
    // ball[X] = clamp(ball[X] + coords[0], minx + r, maxx - r);
    // ball[Y] = clamp(ball[Y] + coords[1], miny + r, maxy - r);
    ball[X] += coords[0];
    ball[Y] += coords[1];
}


__global__
void solve(float* balls, const int count,
           const float gravity,
           const float minx, const float maxx,
           const float miny, const float maxy)
{
    int idx = blockIdx.x * BLOCK + threadIdx.x;

    if (idx >= count)
        return;

    float* ball = balls + idx * SIZE;
    apply_forces(ball, gravity);

    __syncthreads();

    for (int j = 0; j < ITERATIONS; j++)
    {
        solve_collisions(balls, count, minx, maxx, miny, maxy);
        __syncthreads();
    }
}


void solve_gpu(float* balls, const int count,
              const float gravity,
              const float minx, const float maxx,
              const float miny, const float maxy)
{
    solve<<<count, BLOCK>>>(balls, count, gravity, minx, maxx, miny, maxy);
}

