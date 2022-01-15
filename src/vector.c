

#include "vector.h"

#include <string.h>
#include <assert.h>
#include <stdlib.h>


void vector_init(vector_t* vec, size_t elem_size)
{
    memset(vec, 0, sizeof(vector_t));
    vec->elem_size = elem_size;
    vec->data = malloc(elem_size);
    vec->reserved = 1;
}


void vector_push_back(vector_t* vec, const void* elem, size_t elem_size)
{
    assert(elem_size == vec->elem_size);

    while (vec->count + 1 >= vec->reserved)
    {
        vec->reserved *= 2;
        vec->data = realloc(vec->data, vec->reserved * vec->elem_size);
    }

    memcpy((char*)vec->data + vec->count * elem_size, elem, elem_size);
    vec->count++;
}


void* vector_at(vector_t* vec, size_t idx)
{
    return ((char*) vec->data) + idx * vec->elem_size;
}


void vector_destroy(vector_t* vec)
{
    free(vec->data);
}
