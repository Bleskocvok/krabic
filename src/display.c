
#include "display.h"


#include <string.h>
#include <math.h>

#include <SDL2/SDL.h>


#define UNUSED(x) ((void) x);


struct display_t_
{
    SDL_Window* win;
    SDL_Renderer* renderer;
};


struct tex_t_
{
    SDL_Texture* texture;
};


static SDL_Surface* create_surface(int width, int height)
{
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        return SDL_CreateRGBSurface(0, width, height, 32,
                0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff);
#else
        return SDL_CreateRGBSurface(0, width, height, 32,
                0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
#endif
}


static void set_pixel(SDL_Surface* surface, int x, int y,
                      int r, int g, int b, int a)
{
    ((Uint32*)surface->pixels)[x + y * surface->w]
                = SDL_MapRGBA(surface->format, r, g, b, a);
}


static SDL_Surface* make_circle(int size)
{
    SDL_Surface* surface = create_surface(size, size);

    SDL_LockSurface(surface);

    float middle = size * 0.5f;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            float dx = x - middle;
            float dy = y - middle;

            if (dx * dx + dy * dy <= size * size * 0.25f)
            {
                set_pixel(surface, x, y, 255, 255, 255, 255);
            }
            else
            {
                set_pixel(surface, x, y, 0, 0, 0, 0);
            }
        }
    }

    SDL_UnlockSurface(surface);

    return surface;
}



int display_make(display_t** result, int w, int h, const char* title)
{
    *result = calloc(1, sizeof(display_t));
    display_t* display = *result;

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;

    int xy = SDL_WINDOWPOS_UNDEFINED;
    if ((display->win = SDL_CreateWindow(title, xy, xy, w, h, 0)) == NULL)
        return 1;

    Uint32 fl = SDL_RENDERER_PRESENTVSYNC;
    if ((display->renderer = SDL_CreateRenderer(display->win, -1, fl)) == NULL)
        return 1;

    SDL_SetRenderDrawColor(display->renderer, 0, 0, 0, 255);

    return 0;
}


int display_destroy(display_t* display)
{
    SDL_DestroyRenderer(display->renderer);
    SDL_DestroyWindow(display->win);
    SDL_Quit();
    return 0;
}


// int texture_make(tex_t** tex,
//                  display_t* display,
//                  const char* filename)
// {
//     *tex = calloc(1, sizeof(tex_t));
//     // TODO
//     return 0;
// }


int texture_make_circle(tex_t** tex, display_t* display, int size)
{
    *tex = calloc(1, sizeof(tex_t));

    SDL_Surface* circle = make_circle(size);

    (*tex)->texture = SDL_CreateTextureFromSurface(display->renderer, circle);

    SDL_FreeSurface(circle);

    return 0;
}


void texture_destroy(tex_t* tex)
{
    SDL_DestroyTexture(tex->texture);
    memset(tex, 0, sizeof(tex_t));
}


void texture_draw(display_t* display,
                  tex_t* tex,
                  float radius,
                  float x,
                  float y)
{
    SDL_Rect rect;
    rect.x = round(x - radius / 2);
    rect.y = round(y - radius / 2);
    rect.w = round(radius);
    rect.h = round(radius);
    SDL_RenderCopy(display->renderer, tex->texture, NULL, &rect);
}


void display_clear(display_t* display)
{
    SDL_RenderClear(display->renderer);
}


void display_render(display_t* display)
{
    SDL_RenderPresent(display->renderer);
}


const char* get_error(display_t* display)
{
    UNUSED(display);
    return SDL_GetError();
}


int has_quit(display_t* display)
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
        {
            return 1;
        }
        if (event.type == SDL_KEYDOWN)
        {
            if (event.key.keysym.sym == SDLK_ESCAPE)
                return 1;
        }
    }

    UNUSED(display);

    return 0;
}
