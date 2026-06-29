#include "tensor.h"
#include "ops.h"

int main() {
    int w1_shape[] = {784, 128};
    int w2_shape[] = {128,10};
    int b1_shape[] = {128};
    int b2_shape[] = {10};
    tensor w1 = tensor_load("w1.bin", 2, w1_shape);
    tensor w2 = tensor_load("w2.bin", 2, w2_shape);
    tensor b1 = tensor_load("b1.bin",1, b1_shape);
    tensor b2 = tensor_load("b2.bin", 1, b2_shape);

    //load test image.bin
    int image_shape[] = {1, 784};
    tensor input = tensor_load("image.bin", 2, image_shape);

    //inference
    tensor hidden = matmul(&input, &w1);
    add_bias(&hidden, &b1);
    relu(&hidden);
    tensor output = matmul(&hidden, &w2);
    add_bias(&output, &b2);
    softmax(&output);

    tensor_print(&output);
    int max_idx = 0;
    float max = output.data[0];
    for (int i =0; i<output.size; i++) {
        if (max < output.data[i]) {
            max = output.data[i];
            max_idx = i;
        }
    }
    printf("predicted digit: %d and value: %f", max_idx, max);


    //free 
    tensor_free(&w1);
    tensor_free(&w2);
    tensor_free(&b1);
    tensor_free(&b2);
    tensor_free(&input);
    tensor_free(&hidden);
    tensor_free(&output);


}