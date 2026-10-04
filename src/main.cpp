#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tinyml/device_buffer.hpp"
#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"
int main()
{
    sycl::queue queue{sycl::gpu_selector_v};
    //sycl::queue queue{sycl::gpu_selector_v,sycl::property::queue::enable_profiling{}};
    //sycl::queue queue{sycl::gpu_selector_v,{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};

    std::cout << "Device: "
              << queue.get_device()
                     .get_info<sycl::info::device::name>()
              << "\n";

    
    return 0;
}