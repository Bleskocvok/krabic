
#define BLOCK 128


__device__
void solve(float* balls, const int count,
           const float gravity,
           const float minx, const float maxx,
           const float miny, const float maxy)
{
    int idx = blockIdx.x * BLOCK + threadIdx.x;

    if (idx >= count)
        return;

    // TODO
}


void solve_gpu(float* balls, const int count,
              const float gravity,
              const float minx, const float maxx,
              const float miny, const float maxy)
{
    solve<<<count, BLOCK>>>(balls, count, gravity, minx, maxx, miny, maxy);
}

