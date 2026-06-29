#include "tensor.h" 

int main() {
    int shape[] = {5, 2};
    tensor t = tensor_create(shape, 2);
    int idx[2];
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            idx[0] = i;
            idx[1] = j;
            tensor_set(&t, idx, 1.0f);  
        }
    }
    tensor_print(&t);
    tensor_free(&t);
    tensor_print(&t);
    return 0;
}