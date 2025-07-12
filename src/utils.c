//
// Created by Marco  on 11/07/25.
//
#include "utils.h"
#include <math.h>

/* ReLU activation function */
float relu(float x) {
    return x > 0 ? x : 0;
}

/* Derivative of ReLU activation function */
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/* Apply softmax activation function to an array input, and
 * set the result into output. */
void softmax(float *input, float *output, int size) {
    /* Find maximum value then subtact it to avoid
     * numerical stability issues with exp(). */
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Calculate exp(x_i - max) for each element and sum.
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities.
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    } else {
        /* Fallback in case of numerical issues, just provide
         * a uniform distribution. */
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / size;
        }
    }
}
