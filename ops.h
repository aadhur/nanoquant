#ifndef OPS_H
#define OPS_H

#include "tensor.h"

tensor matmul(tensor* A, tensor* B);
void relu(tensor* T);
void softmax(tensor* T);
void add_bias(tensor* T, tensor* bias);
void update_minmax(tensor* T, float& min, float& max);
#endif