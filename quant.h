// quant.h
#ifndef QUANT_H
#define QUANT_H

#include "tensor.h"
#include <stdint.h>

qtensor quantize (tensor* T, float scale);
tensor dequantize(qtensor* T, float scale);
float get_scale(float min_X, float max_X);
tensor qmatmul(qtensor* A, qtensor* B, float scale_A, float scale_B);
#endif