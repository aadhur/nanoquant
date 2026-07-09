#include "quant.h"
#include <cmath>
#include <algorithm>
#include "tensor.h"

float get_scale(float min_X, float max_X) {
    float abs_max = std::max(std::abs(min_X), std::abs(max_X));
    return abs_max/127.0f;
}

qtensor quantize (tensor* T, float scale){
    qtensor qt = qtensor_create(T->shape, T-> ndims);
    for (int i =0; i<T->size; i++) {
        int q = round(T->data[i]/scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qt.data[i] = (int8_t)q;
    }
    return qt;
}
tensor dequantize(qtensor* T, float scale){
    tensor t = tensor_create(T->shape, T->ndims);
    for (int i =0; i<T->size; i++) {
        t.data[i] = scale * T->data[i];
    }
    return t;
}

tensor qmatmul(qtensor* A, qtensor* B, float scale_A, float scale_B) {
    if (A->ndims != 2 || B->ndims != 2) {
        printf("qmatmul only supports 2D tensors for now\n");
        exit(1);
    }
    if (A->shape[1] != B->shape[0]) {
        printf("qmatmul shape mismatch\n");
        exit(1);
    }

    int out_shape[] = {A->shape[0], B->shape[1]};
    tensor out = tensor_create(out_shape, 2);   // output is float, not qtensor

    for (int i = 0; i < A->shape[0]; i++) {
        for (int j = 0; j < B->shape[1]; j++) {
            int32_t sum = 0;   // accumulate in int32, not float
            for (int k = 0; k < A->shape[1]; k++) {
                int flat_A = i * A->shape[1] + k;
                int flat_B = k * B->shape[1] + j;
                sum += (int32_t)A->data[flat_A] * (int32_t)B->data[flat_B];
            }
            int flat_out = i * B->shape[1] + j;
            out.data[flat_out] = sum * scale_A * scale_B;  // rescale back to real float value
        }
    }
    return out;
}