
#include "display.h"
#include "physics.h"
#include "vector.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void mk_ball(float* ball, float x, float y, float r)
{
    ball[X] = x;
    ball[Y] = y;
    ball[PREV_X] = x;
    ball[PREV_Y] = y;
    ball[RADIUS] = r;
}


int main()
{
    vector_t balls;
    vector_init(&balls, SIZE * sizeof(float));

    const int screen_w = 600;
    const int screen_h = 600;
    const float gravity = 10;
    const float radius = 4;
    const int ball_count = 2000;

    srand(1337);
    for (int i = 0; i < ball_count; i++)
    {
        float ball[SIZE] = { 0 };
        mk_ball(ball, rand() % screen_w, rand() % screen_h, radius);
        PUSH_STRUCT(&balls, &ball);
    }

    display_t* display;

    if (display_make(&display, screen_w, screen_h, "thing") != 0)
        return fprintf(stderr, "error: %s\n", get_error(display)),
               EXIT_FAILURE;

    tex_t* tex;

    texture_make_circle(&tex, display, 64);

    if (physics_init() != 0)
        return EXIT_FAILURE;

    // wait for ESCAPE
    while (!has_quit(display))
    {  }

    while (!has_quit(display))
    {
        solve_circles(balls.data, balls.count, gravity,
                      0, screen_w, 0, screen_h);

        display_clear(display);

        for (size_t i = 0; i < balls.count; i++)
        {
            float* b = vector_at(&balls, i);
            texture_draw(display, tex, b[RADIUS] * 2, b[X], b[Y]);
        }

        display_render(display);
    }

    physics_quit();

    vector_destroy(&balls);

    texture_destroy(tex);
    display_destroy(display);

    return EXIT_SUCCESS;
}
