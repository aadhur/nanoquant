#include "ops.h"
#include <math.h>

tensor matmul(tensor* A, tensor* B) {
    if (A->ndims != 2 || B->ndims != 2) {
        printf("matmul only supports 2D tensors for now\n");
        exit(1);
    }
    if (A->shape[1] != B->shape[0]) {
        printf("matmul shape mismatch\n");
        exit(1);
    }

    int out_shape[] = {A->shape[0], B->shape[1]};
    tensor out = tensor_create(out_shape, 2);

    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < B->shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->shape[1]; k++) {
                int flat_A = i * A->shape[1] + k;
                int flat_B = k * B->shape[1] + j;
                sum += A->data[flat_A] * B->data[flat_B];
            }
            int flat_out = i * B->shape[1] + j;
            out.data[flat_out] = sum;
        }
    }
    return out;
}

void relu(tensor* T) {
    for (int i=0;i<T->size;i++) {
        if (T->data[i] < 0) {
            T->data[i] = 0;
        }
    }
}

void softmax(tensor* T) {
    float max = T->data[0];
    for (int i=0; i< T->size;i++) {
        if (max < T->data[i]) {
            max = T->data[i];
        }
    }
    float sum = 0;
    // step 2 - exp each element and sum
    float sum = 0;
    for (int i = 0; i < T->size; i++) {
        T->data[i] = exp(T->data[i] - max); 
        sum += T->data[i];
    }

    // step 3 -  divide by sum
    for (int i = 0; i < T->size; i++) {
        T->data[i] = T->data[i] / sum;
    }

}
