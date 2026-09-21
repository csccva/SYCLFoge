#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>

# include "tinyml/device_buffer.hpp"
# include "tinyml/tensor.hpp"
# include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"

int main()
{
    sycl::queue queue{sycl::gpu_selector_v};

    tinyml::Tensor<float> input(queue, {6});
    tinyml::Tensor<float> activation(queue, {6});

    std::vector<float> h_input = {
        -3.0f, -1.0f, 0.0f, 2.0f, 5.0f, -4.0f
    };

    sycl::event e_copy =
        ops::copy_to_device(queue, input, h_input.data());

    sycl::event e_relu =
        ops::relu(queue, input, activation, 256, {e_copy});

    std::vector<float> h_output(6);

    ops::copy_to_host(
        queue, h_output.data(), activation, {e_relu}
    ).wait();

    for (float value : h_output) {
        std::cout << value << " ";
    }

    std::cout << std::endl;

    return 0;
}