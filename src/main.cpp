#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"

int main()
{
    sycl::queue queue{sycl::gpu_selector_v};

    // Network:
    // X [3,2] -> Linear(2,4) -> H1 [3,4]
    //         -> ReLU        -> H2 [3,4]
    //         -> Linear(4,2) -> Y  [3,2]

    tinyml::Linear<float> linear1(queue, 2, 4);
    tinyml::Linear<float> linear2(queue, 4, 2);

    tinyml::Tensor<float> X(queue,  {3,2});
    tinyml::Tensor<float> H1(queue, {3,4});
    tinyml::Tensor<float> H2(queue, {3,4});
    tinyml::Tensor<float> Y(queue,  {3,2});

    // Input
    std::vector<float> h_X{
        3.2f, 1.5f,
        3.2f, 1.5f,
        3.2f, 1.5f
    };

    sycl::event e_X = ops::copy_to_device(queue, X, h_X.data());

    // Linear 1 parameters: W1 [2,4], b1 [4]
    std::vector<float> h_W1{
         2.1f,  3.4f, 1.01f, 0.1f,
        -0.1f, -0.3f, 0.88f, 2.7f
    };

    std::vector<float> h_b1{
        0.1f, -0.37f, 0.1f, -7.5f
    };

    sycl::event e_W1 = ops::copy_to_device(queue, linear1.weight(), h_W1.data());

    sycl::event e_b1 = ops::copy_to_device(queue, linear1.bias(), h_b1.data());

    // First linear layer
    sycl::event e_l1 =
        linear1.forward(
            X, H1,
            256, 16,
            {e_X, e_W1, e_b1}
        );

    // ReLU
    sycl::event e_relu1 =
        ops::relu(queue, H1, H2, 256, {e_l1});

    // Linear 2 parameters: W2 [4,2], b2 [2]
    std::vector<float> h_W2{
         0.1f,  -0.4f,
         1.31f,  0.91f,
        -0.1f,  -0.3f,
        -0.65f, -2.0f
    };

    std::vector<float> h_b2{
        -0.95f, -9.0f
    };

    sycl::event e_W2 = ops::copy_to_device(queue, linear2.weight(), h_W2.data());

    sycl::event e_b2 = ops::copy_to_device(queue, linear2.bias(), h_b2.data());

    // Second linear layer
    sycl::event e_l2 = linear2.forward(H2, Y,256, 16,{e_relu1, e_W2, e_b2});

    // Copy final result back to CPU
    std::vector<float> h_Y(6);

    ops::copy_to_host(queue,h_Y.data(),Y,{e_l2}).wait();

    for (float value : h_Y) {
        std::cout << value << " ";
    }

    std::cout << std::endl;

    {std::vector<float> h_in{
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };
    tinyml::Tensor<float> input(queue,{3,4});
    sycl::event e_i=ops::copy_to_device(queue,input,h_in.data());
    tinyml::Tensor<float> output(queue,{3,4});
    sycl::event e_sm=ops::softmax(queue,input,output,256,{e_i});
    std::vector<float> h_ou(12);
    sycl::event e_ou=ops::copy_to_host(queue,h_ou.data(),output,{e_sm});
    e_ou.wait();
    for(float value : h_ou){
        std::cout << value <<  " ";
    }
    std::cout << std::endl;}
    {

        std::cout << std::endl;
    
        std::vector<float> h_in{
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            4.0f, 3.0f, 2.0f, 1.0f
        };
        tinyml::Tensor<float> input(queue,{3,4});
        sycl::event e_i=ops::copy_to_device(queue,input,h_in.data());
        tinyml::Tensor<float> output(queue,{3,4});
        sycl::event e_sm=ops::softmax_parallel(queue,input,output,256,{e_i});
        std::vector<float> h_ou(12);
        sycl::event e_ou=ops::copy_to_host(queue,h_ou.data(),output,{e_sm});
        e_ou.wait();
        for(float value : h_ou){
            std::cout << value <<  " ";
        }
        std::cout << std::endl;}
        {
            {
                const std::size_t M = 3;
                const std::size_t N = 1000;
                const std::size_t group_size = 256;
            
                // Create some non-trivial input
                std::vector<float> h_in(M * N);
            
                for (std::size_t i = 0; i < M; i++) {
                    for (std::size_t j = 0; j < N; j++) {
                        h_in[i * N + j] =
                            static_cast<float>(j % 17) * 0.1f
                            - static_cast<float>(i);
                    }
                }
            
                tinyml::Tensor<float> input(queue, {M, N});
                tinyml::Tensor<float> output_serial(queue, {M, N});
                tinyml::Tensor<float> output_parallel(queue, {M, N});
            
                // Copy input
                sycl::event e_input =
                    ops::copy_to_device(queue, input, h_in.data());
            
                // Reference implementation
                sycl::event e_serial =
                    ops::softmax(
                        queue,
                        input,
                        output_serial,
                        group_size,
                        {e_input});
            
                // Parallel implementation
                sycl::event e_parallel =
                    ops::softmax_parallel(
                        queue,
                        input,
                        output_parallel,
                        group_size,
                        {e_input});
            
                // Copy both results back
                std::vector<float> h_serial(M * N);
                std::vector<float> h_parallel(M * N);
            
                sycl::event e_serial_copy =
                    ops::copy_to_host(
                        queue,
                        h_serial.data(),
                        output_serial,
                        {e_serial});
            
                sycl::event e_parallel_copy =
                    ops::copy_to_host(
                        queue,
                        h_parallel.data(),
                        output_parallel,
                        {e_parallel});
            
                e_serial_copy.wait();
                e_parallel_copy.wait();
            
                // Compare serial and parallel versions
                float max_diff = 0.0f;
            
                for (std::size_t i = 0; i < M * N; i++) {
                    float diff = std::abs(h_serial[i] - h_parallel[i]);
            
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
            
                std::cout << "Maximum difference: "
                          << max_diff << std::endl;
            
                // Check that every parallel softmax row sums to ~1
                for (std::size_t i = 0; i < M; i++) {
            
                    float row_sum = 0.0f;
            
                    for (std::size_t j = 0; j < N; j++) {
                        row_sum += h_parallel[i * N + j];
                    }
            
                    std::cout << "Row " << i
                              << " sum: " << row_sum
                              << std::endl;
                }
            }
        }
    return 0;
}