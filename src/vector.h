
#pragma once

#include <stdlib.h>


#define PUSH_STRUCT(vec, elem) \
        vector_push_back(vec, elem, sizeof(*elem))


typedef struct
{
    void* data;

    size_t elem_size;

    size_t count;

    size_t reserved;

} vector_t;

void vector_init(vector_t* vec, size_t elem_size);

void vector_push_back(vector_t* vec, const void* elem, size_t elem_size);

void* vector_at(vector_t* vec, size_t idx);

void vector_destroy(vector_t* vec);
