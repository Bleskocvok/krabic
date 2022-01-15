
#pragma once


struct display_t_;
typedef struct display_t_ display_t;

struct tex_t_;
typedef struct tex_t_ tex_t;


int display_make(display_t** display, int w, int h, const char* title);

int display_destroy(display_t* display);

// int texture_make(tex_t** tex,
//                  display_t* display,
//                  const char* filename);

int texture_make_circle(tex_t** tex, display_t* display, int size);

void texture_destroy(tex_t* tex);

void texture_draw(display_t* display,
                 tex_t* tex,
                 float radius,
                 float x,
                 float y);

void display_clear(display_t* display);

void display_render(display_t* display);

const char* get_error(display_t* display);

int has_quit(display_t* display);
