#include "ops.h"
#include <math.h>

tensor matmul(tensor* A, tensor* B) {
    if (A->ndims != 2 || B->ndims != 2) {
        printf("matmul only supports 2D tensors for now\n");
        exit(1);
    }
    if (A->shape[1] != B->shape[0]) { //dimension check for mat mul
        printf("matmul shape mismatch\n");
        exit(1);
    }

    int out_shape[] = {A->shape[0], B->shape[1]}; //final dimensions
    tensor out = tensor_create(out_shape, 2); //empty tensor

    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < B->shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->shape[1]; k++) {
                int flat_A = i * A->shape[1] + k; //flat index for A
                int flat_B = k * B->shape[1] + j; //flat index for B (k is both the col for A and row for B)
                sum += A->data[flat_A] * B->data[flat_B];
            }
            int flat_out = i * B->shape[1] + j; 
            out.data[flat_out] = sum;
        }
    }
    return out;
}

void relu(tensor* T) { //if less than 0 -> 0
    for (int i=0;i<T->size;i++) {
        if (T->data[i] < 0) {
            T->data[i] = 0;
        }
    }
}

void softmax(tensor* T) {
    float max = T->data[0];
    for (int i=0; i< T->size;i++) { //get the max val
        if (max < T->data[i]) {
            max = T->data[i];
        }
    }
    float sum = 0;
    // step 2 - exp each element and sum
    for (int i = 0; i < T->size; i++) { 
        T->data[i] = exp(T->data[i] - max); 
        sum += T->data[i];
    }

    // step 3 -  divide by sum
    for (int i = 0; i < T->size; i++) {
        T->data[i] = T->data[i] / sum;
    }

}

void add_bias(tensor* T, tensor* bias) {
    for (int i = 0; i < T->shape[0]; i++) {
        for (int j = 0; j < T->shape[1]; j++) {
            int flat = i * T->shape[1] + j;
            T->data[flat] += bias->data[j];
        }
    }
}
