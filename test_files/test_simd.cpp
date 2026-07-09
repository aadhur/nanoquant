#include "../tensor.h"
#include "../ops.h"
#include <string>
#include <fstream>

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
    std::ifstream label_file("calib_images/labels.txt");
    std::string fname;
    int true_label;
    int correct = 0;

    float min_hidden = 1e9f,  max_hidden = -1e9f;
    float min_relu = 1e9f, max_relu = -1e9f;
    float min_output = 1e9f, max_output = -1e9f;

    for (int i = 100; i < 300; i++) {
        std::string filename = "calib_images/image_" + std::to_string(i) + ".bin";
        tensor input = tensor_load(filename.c_str(), 2, image_shape);
        // tensor input = tensor_load("calib_images/image_0.bin", 2, image_shape);

        tensor hidden = matmul(&input, &w1);
        add_bias(&hidden, &b1);

        relu(&hidden);

        tensor output = matmul(&hidden, &w2);
        add_bias(&output, &b2);
        
        //checking output[10] tensor struct
        // printf("output.size = %d\n", output.size);
        // for (int k = 0; k < output.size; k++) {
        //     printf("output[%d] = %f\n", k, output.data[k]);
        // }


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
        tensor_free(&hidden);
        tensor_free(&output);
    }

    printf("Accuracy: %d/200\n", correct);
    // printf("hidden (pre-relu) range: [%f, %f]\n", min_hidden, max_hidden);
    // printf("relu output range: [%f, %f]\n", min_relu, max_relu);
    // printf("output (pre-softmax) range: [%f, %f]\n", min_output, max_output);
    //free 
    tensor_free(&w1);
    tensor_free(&w2);
    tensor_free(&b1);
    tensor_free(&b2);



}