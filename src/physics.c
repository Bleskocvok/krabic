
#include "physics.h"

#include <math.h>
#include <stdio.h>


int physics_init()
{
    printf("(CPU) Initialized physics\n");
    return 0;
}


void physics_quit()
{  }


static float clamp(float val, float l, float h)
{
    if (val < l)
        return l;
    if (val > h)
        return h;
    return val;
}


static void apply_forces(float* ball, float gravity)
{
    float dx = ball[X] - ball[PREV_X];
    float dy = ball[Y] - ball[PREV_Y];

    ball[PREV_X] = ball[X];
    ball[PREV_Y] = ball[Y];

    ball[X] += dx;
    ball[Y] += dy;

    ball[Y] += gravity * 0.003;
}


static void collision(const float* a, const float* b, float* result)
{
    float dx = a[X] - b[X];
    float dy = a[Y] - b[Y];

    float d = a[RADIUS] + b[RADIUS];

    if (dx * dx + dy * dy >= d * d)
        return;

    float dist = sqrt(dx * dx + dy * dy);

    float diff_x = (dist - d) / dist;
    float diff_y = (dist - d) / dist;

    result[0] -= dx * diff_x * 0.5;
    result[1] -= dy * diff_y * 0.5;
}


static void solve_collisions(float* balls, int count,
                             float minx, float maxx,
                             float miny, float maxy)
{
    float coords[count * 2];

    for (int i = 0; i < count; i++)
    {
        coords[i * 2] = 0;
        coords[i * 2 + 1] = 0;
    }

    for (int b = 0; b < count; b++)
    {
        float* ball = balls + b * SIZE;

        for (int i = 0; i < count; i++)
        {
            if (b != i)
            {
                collision(ball, balls + i * SIZE, coords + i * 2);
            }
        }
    }

    for (int i = 0; i < count; i++)
    {
        float* ball = balls + i * SIZE;
        float r = ball[RADIUS];

        ball[X] = clamp(ball[X] - coords[i * 2],     minx + r, maxx - r);
        ball[Y] = clamp(ball[Y] - coords[i * 2 + 1], miny + r, maxy - r);
    }
}


void solve_circles(float* balls, int count,
                   float gravity,
                   float minx, float maxx,
                   float miny, float maxy)
{
    for (int i = 0; i < count; i++)
    {
        float* ball = balls + i * SIZE;
        
        apply_forces(ball, gravity);
    }

    for (int j = 0; j < ITERATIONS; j++)
        solve_collisions(balls, count, minx, maxx, miny, maxy);
}
