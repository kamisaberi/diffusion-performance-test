#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <numeric>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// Helper code (Logger, GpuTimer, CHECK_CUDA, etc. from previous examples)
// ... Omitted for brevity, but required ...
class Logger : public nvinfer1::ILogger { /* ... */ };
class GpuTimer { /* ... */ };
#define CHECK_CUDA(call) { /* ... */ }
#define CHECK_CURAND(call) { /* ... */ }

const int NUM_ITERATIONS = 10;
const int WARMUP_ITERATIONS = 2;
const int BATCH_SIZE = 1;
const int TIMESTEPS = 50;
const int IMG_SIZE = 64;

// Simple CUDA kernel for the sampling math
__global__ void sampling_kernel(float* img, const float* predicted_noise, const float* random_noise,
                                float alpha_t, float alpha_t_cumprod, float beta_t, int num_elements) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        float inv_sqrt_alpha_t = rsqrtf(alpha_t);
        float noise_coeff = (1.0f - alpha_t) / sqrtf(1.0f - alpha_t_cumprod);
        img[i] = inv_sqrt_alpha_t * (img[i] - noise_coeff * predicted_noise[i]) + sqrtf(beta_t) * random_noise[i];
    }
}

int main(int argc, char** argv[]) {
    // ... Boilerplate for building/loading TensorRT engine (same as last time) ...
    // ... This part is long and unchanged from the GAN example ...
    // --- The key difference is the main benchmark loop ---

    std::cout << "Starting benchmark...\n";
    GpuTimer timer;
    float total_time = 0;

    // Allocate a buffer for random noise for the sampling step
    float* d_random_noise;
    size_t num_elements = (size_t)BATCH_SIZE * 3 * IMG_SIZE * IMG_SIZE;
    CHECK_CUDA(cudaMalloc(&d_random_noise, num_elements * sizeof(float)));

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        
        // Initial image is pure noise
        CHECK_CURAND(curandGenerateNormal(gen, (float*)buffers[output_idx], num_elements, 0.0f, 1.0f));

        for (int t_idx = TIMESTEPS - 1; t_idx >= 0; --t_idx) {
            // The output of the previous step becomes the input of the next
            std::swap(buffers[input_idx], buffers[output_idx]);

            // Set the time step input
            h_time[0] = static_cast<float>(t_idx);
            CHECK_CUDA(cudaMemcpyAsync(d_time, h_time.data(), sizeof(float), cudaMemcpyHostToDevice, stream));
            
            // Run the U-Net with TensorRT
            context->enqueueV3(stream);

            // Generate noise for the sampling step
            if (t_idx > 0) {
                CHECK_CURAND(curandGenerateNormal(gen, d_random_noise, num_elements, 0.0f, 1.0f));
            } else {
                CHECK_CUDA(cudaMemsetAsync(d_random_noise, 0, num_elements * sizeof(float), stream));
            }
            
            // Run the custom sampling kernel
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            sampling_kernel<<<blocks, threads, 0, stream>>>(
                (float*)buffers[input_idx], (float*)buffers[output_idx], d_random_noise,
                h_alphas[t_idx], h_alphas_cumprod[t_idx], h_betas[t_idx], num_elements
            );
        }
        
        timer.stop();
        total_time += timer.elapsed_ms();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // ... Print results and cleanup ...
    return 0;
}