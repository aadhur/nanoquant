// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <stdint.h>


typedef struct tensor {
    float* data;
    int* shape; //(rows, cols, channels, batches)
    int size;
    int ndims;
} tensor;

typedef struct qtensor {
    int8_t* data;
    int* shape;
    int size;
    int ndims;
} qtensor;

tensor tensor_create( int* shape, int ndims);
qtensor qtensor_create(int *shape, int ndims);
tensor tensor_load(const char* filepath, int ndims, int *shape);
void tensor_free(tensor* T);
void qtensor_free(qtensor* T);
float tensor_get(tensor* T, int* indices);
void tensor_set(tensor* T, int* indices, float val);
void tensor_print(tensor* T);


#endif
