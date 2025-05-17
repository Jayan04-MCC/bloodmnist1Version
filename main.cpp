
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_SAMPLES 100
#define INPUT_SIZE 784
#define OUTPUT_SIZE 8

__device__ void softmax(float* input, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

__global__ void infer_batch(const float* inputs, const float* weights, const float* biases, int* predictions) {
    int sample_idx = blockIdx.x;

    float logits[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += inputs[sample_idx * INPUT_SIZE + i] * weights[j * INPUT_SIZE + i];
        }
        logits[j] = sum + biases[j];
    }

    softmax(logits, OUTPUT_SIZE);

    // Obtener la clase con mayor probabilidad
    float max_val = logits[0];
    int max_idx = 0;
    for (int j = 1; j < OUTPUT_SIZE; j++) {
        if (logits[j] > max_val) {
            max_val = logits[j];
            max_idx = j;
        }
    }

    predictions[sample_idx] = max_idx;
}


int main() {
    size_t F = 100; //dataset de 100 vectores
    size_t C = 784; // tamanio de las imagenes
    int* matrixCpu;
    int* matrixGpu;
    matrixCpu=(int*)malloc(F * C);

    cudaMalloc(&matrixGpu, F * C);
}