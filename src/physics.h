
#pragma once


#define X 0
#define Y 1
#define PREV_X 2
#define PREV_Y 3
#define RADIUS 4
#define SIZE 5


int physics_init();

void physics_quit();

void solve_circles(float* balls, int count,
                   float gravity,
                   float minx, float maxx,
                   float miny, float maxy);
