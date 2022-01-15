
#include "display.h"
#include "physics.h"
#include "vector.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>



typedef struct
{
    float x,
          y,
          prev_x,
          prev_y,
          radius;

} ball_t;


ball_t mk_ball(float x, float y, float r)
{
    return (ball_t)
    {
        .x = x,
        .y = y,
        .prev_x = x,
        .prev_y = y,
        .radius = r,
    };
}


int main()
{
    vector_t balls;
    vector_init(&balls, sizeof(ball_t));

    const int screen_w = 600;
    const int screen_h = 600;
    const float gravity = 10;
    const int ball_count = 200;

    srand(1337);
    for (int i = 0; i < ball_count; i++)
    {
        ball_t b = mk_ball(rand() % screen_w, rand() % screen_h, 10);
        PUSH_STRUCT(&balls, &b);
    }


    display_t* display;

    if (display_make(&display, screen_w, screen_h, "thing") != 0)
        return fprintf(stderr, "error: %s\n", get_error(display)),
               EXIT_FAILURE;

    tex_t* tex;

    texture_make_circle(&tex, display, 64);

    if (physics_init() != 0)
        return EXIT_FAILURE;

    while (!has_quit(display))
    {
        solve_circles(balls.data, balls.count, gravity,
                      0, screen_w, 0, screen_h);

        display_clear(display);

        for (size_t i = 0; i < balls.count; i++)
        {
            ball_t b = *(ball_t*)vector_at(&balls, i);
            texture_draw(display, tex, b.radius * 2, b.x, b.y);
        }

        display_render(display);
    }

    physics_quit();

    vector_destroy(&balls);

    texture_destroy(tex);
    display_destroy(display);

    return EXIT_SUCCESS;
}
