
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <c10/cuda/CUDAStream.h>

const int NUM_ITERATIONS = 10;
const int WARMUP_ITERATIONS = 2;
const int BATCH_SIZE = 1;
const int TIMESTEPS = 50;
const int IMG_SIZE = 64;

int main(int argc, const char* argv[]) {
    std::cout << "--- Performance Test: Diffusion with C++/LibTorch ---" << std::endl;
    if (argc != 2) { std::cerr << "Usage: ./libtorch_infer <path_to_model.pt>\n"; return -1; }
    if (!torch::cuda::is_available()) { std::cerr << "CUDA is not available.\n"; return -1; }

    torch::Device device(torch::kCUDA);
    torch::jit::Module module;
    try { module = torch::jit::load(argv[1]); }
    catch (const c10::Error& e) { std::cerr << "Error loading model: " << e.what() << "\n"; return -1; }
    module.to(device);
    module.eval();

    auto betas = torch::linspace(0.0001, 0.02, TIMESTEPS, device);
    auto alphas = 1. - betas;
    auto alphas_cumprod = torch::cumprod(alphas, 0);

    std::cout << "Warming up...\n";
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            auto img = torch::randn({BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE}, device);
            for (int t_idx = TIMESTEPS - 1; t_idx >= 0; --t_idx) {
                auto t = torch::full({BATCH_SIZE}, static_cast<float>(t_idx), device).to(torch::kFloat);
                std::vector<torch::jit::IValue> inputs = {img, t};
                auto predicted_noise = module.forward(inputs).toTensor();
            }
        }
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
    
    std::cout << "Starting benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            auto img = torch::randn({BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE}, device);
            for (int t_idx = TIMESTEPS - 1; t_idx >= 0; --t_idx) {
                auto t = torch::full({BATCH_SIZE}, static_cast<float>(t_idx), device).to(torch::kFloat);
                std::vector<torch::jit::IValue> inputs = {img, t};
                auto predicted_noise = module.forward(inputs).toTensor();
                
                auto alpha_t = alphas[t_idx];
                auto alpha_t_cumprod = alphas_cumprod[t_idx];
                auto beta_t = betas[t_idx];
                auto noise = (t_idx > 0) ? torch::randn_like(img) : torch::zeros_like(img);

                img = (1 / torch::sqrt(alpha_t)) * (img - ((1 - alpha_t) / (torch::sqrt(1 - alpha_t_cumprod))) * predicted_noise) + torch::sqrt(beta_t) * noise;
            }
        }
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double avg_time_ms = total_time_ms / NUM_ITERATIONS;

    std::cout << "\n--- LibTorch Results ---\n";
    printf("Total time for %d generations: %.3f ms\n", NUM_ITERATIONS, total_time_ms);
    printf("Average time per generation (%d steps): %.4f ms\n", TIMESTEPS, avg_time_ms);
    return 0;
}