#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <stdint.h>

tensor tensor_create(int* shape, int ndims) {
    tensor t;
    t.ndims = ndims; 
    t.shape = (int*)malloc(ndims * sizeof(int)); //creating a shape 'box' (heap memory) to avoid dangling pointer

    memcpy(t.shape, shape, ndims*sizeof(int)); //values copied, not address perma

    t.size = 1;
    for (int i=0;i< ndims;i++) {
        t.size*=shape[i];
    }
    t.data = (float*)malloc(t.size * sizeof(float)); //heap memory for data for the entire flat array

    return t;

}

qtensor qtensor_create(int* shape, int ndims) {
    qtensor t;
    t.ndims = ndims; 
    t.shape = (int*)malloc(ndims * sizeof(int)); //creating a shape 'box' (heap memory) to avoid dangling pointer

    memcpy(t.shape, shape, ndims*sizeof(int)); //values copied, not address perma

    t.size = 1;
    for (int i=0;i< ndims;i++) {
        t.size*=shape[i];
    }
    t.data = (int8_t*)malloc(t.size * sizeof(int8_t)); //heap memory for data for the entire flat array
    return t;

}

void tensor_free(tensor* T) {
    free(T -> shape);
    free(T -> data);
    T-> shape = NULL;
    T-> data = NULL; //both set to NULL so T knows it's free
}

void qtensor_free(qtensor* T) {
    free(T->data);
    free(T->shape);
    T-> shape = NULL;
    T-> data = NULL;
}

float tensor_get(tensor* T, int* indices) {
    int stride = 1;
    int flat_index = 0;
    for (int i = T->ndims-1; i>=0; i--) { //row major indexing to get flat index - working backwards multiplying stride by each dimension which multiplies with previous index
        flat_index += indices[i] * stride;
        stride *= T->shape[i];
    }
    return T->data[flat_index];
}

void tensor_set(tensor* T, int* indices, float val) {
    int stride = 1;
    int flat_index = 0;
    for (int i = T->ndims-1; i>=0; i--) {
        flat_index += indices[i] * stride;
        stride *= T->shape[i];
    }
    T->data[flat_index] = val;
} 

//specifically for 2D 
// void tensor_print(tensor* T) {
//     int flat_index;
//     float val;
//     for (int i=0;i<T->shape[0];i++) {
//         for(int j=0;j<T->shape[1]; j++) {
//             flat_index = i *T->shape[1] + j;
//             val = T->data[flat_index];
//             printf("%f", val);
//         }
//         printf("\n");
//     }
// }


void tensor_print(tensor* T) {
    for (int i=0; i<T->size;i++) {
        printf("%f ", T->data[i]);
        //to check end of row
        if ((i+1) % T->shape[T->ndims-1] == 0) {
            printf("\n");
        }
    }
}

tensor tensor_load(const char* filepath, int ndims, int *shape) {
    tensor t = tensor_create(shape, ndims);

    //open file
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("File couldn't be opened");
        return t;
    }

    //read file to t.data
    fread(t.data, sizeof(float), t.size, f); //fread(where to put data, size of each item, how many items, file)
    fclose(f);
    return t;
}
