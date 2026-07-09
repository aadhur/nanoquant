#include "../tensor.h"
#include "../ops.h"
#include "../quant.h"
#include <string>
#include <fstream>
#include <chrono>

int main() {
    int w1_shape[] = {784, 128};
    int w2_shape[] = {128,10};
    int b1_shape[] = {128};
    int b2_shape[] = {10};
    tensor w1 = tensor_load("w1.bin", 2, w1_shape);
    tensor w2 = tensor_load("w2.bin", 2, w2_shape);
    tensor b1 = tensor_load("b1.bin",1, b1_shape);
    tensor b2 = tensor_load("b2.bin", 1, b2_shape);

    //check quantize and dequantize on w1 and w2
    float min_w1 = 1e9f,  max_w1 = -1e9f;
    float min_w2 = 1e9f, max_w2 = -1e9f;

    update_minmax(&w1, min_w1, max_w1);
    float scale_w1 = get_scale(min_w1, max_w1);
    qtensor w1q = quantize(&w1, scale_w1);

    update_minmax(&w2, min_w2, max_w2);
    float scale_w2 = get_scale(min_w2, max_w2);
    qtensor w2q = quantize(&w2, scale_w2);

    int image_shape[] = {1, 784};
    std::ifstream label_file("calib_images/labels.txt");
    std::string fname;
    int true_label;
    int correct = 0;

    float min_hidden = 1e9f,  max_hidden = -1e9f;
    float min_relu = 1e9f, max_relu = -1e9f;
    float min_output = 1e9f, max_output = -1e9f;
    float min_input = 1e9f, max_input = -1e9f;

    for(int i=100; i<300; i++) {
        std::string filename = "calib_images/image_" + std::to_string(i) + ".bin";
        tensor input = tensor_load(filename.c_str(), 2, image_shape);
        update_minmax(&input, min_input, max_input);

        tensor hidden = matmul(&input, &w1);

        add_bias(&hidden, &b1);
        update_minmax(&hidden, min_hidden, max_hidden);

        relu(&hidden);
        update_minmax(&hidden, min_relu, max_relu);


        tensor output = matmul(&hidden, &w2);
        add_bias(&output, &b2);
        update_minmax(&output, min_output, max_output);

        tensor_free(&input);
        tensor_free(&hidden);
        tensor_free(&output);

    }
    //get scale
    float scale_input = get_scale(min_input, max_input);
    float scale_hidden = get_scale(min_hidden, max_hidden);
    float scale_relu = get_scale(min_relu, max_relu);
    auto start_int8 = std::chrono::high_resolution_clock::now();
    for (int i = 100; i < 300; i++) {
        std::string filename = "calib_images/image_" + std::to_string(i) + ".bin";
        // tensor input = tensor_load("calib_images/image_0.bin", 2, image_shape);

        tensor input = tensor_load(filename.c_str(), 2, image_shape);
        qtensor input_q = quantize(&input, scale_input);

        tensor hidden = qmatmul(&input_q, &w1q, scale_input, scale_w1);  

        add_bias(&hidden, &b1);

        relu(&hidden);

        qtensor hidden_q2 = quantize(&hidden, scale_relu);

        tensor output = qmatmul(&hidden_q2, &w2q, scale_relu, scale_w2);
        add_bias(&output, &b2);
        softmax(&output);

        int max_idx = 0;
        float max = output.data[0];
        for (int j = 0; j < output.size; j++) {
            if (max < output.data[j]) {
                max = output.data[j];
                max_idx = j;
            }
        }

        label_file >> fname >> true_label;
        if (max_idx == true_label) correct++;

        printf("image %d: predicted %d, actual %d, confidence %f\n", i, max_idx, true_label, max);
        if (max_idx != true_label) {
            printf("Wrong prediciton for %i: %d instead of %d \n", i, max_idx, true_label );
        }
        tensor_free(&input);
        qtensor_free(&input_q);    
        tensor_free(&hidden);
        qtensor_free(&hidden_q2);  
        tensor_free(&output);
    }
    auto end_int8 = std::chrono::high_resolution_clock::now();
    auto duration_int8 = std::chrono::duration_cast<std::chrono::microseconds>(end_int8 - start_int8);
    printf("INT8 total time: %lld microseconds (%.3f us/image)\n", (long long)duration_int8.count(), duration_int8.count() / 200.0);


    // printf("Accuracy: %d/200\n", correct);
    // printf("hidden (pre-relu) range: [%f, %f]\n", min_hidden, max_hidden);
    // printf("relu output range: [%f, %f]\n", min_relu, max_relu);
    // printf("output (pre-softmax) range: [%f, %f]\n", min_output, max_output);
    //free 
    tensor_free(&w1);
    tensor_free(&w2);
    tensor_free(&b1);
    tensor_free(&b2);
    qtensor_free(&w1q);
    qtensor_free(&w2q);

}