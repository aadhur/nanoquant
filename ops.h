#ifndef OPS_H
#define OPS_H

#include "tensor.h"

tensor matmul(tensor* A, tensor* B);
void relu(tensor* T);
void softmax(tensor* T);

#endif