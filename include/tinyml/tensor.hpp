#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include "tinyml/device_buffer.hpp"

namespace tinyml {

template <typename T>
class Tensor {
    public:
        std::size_t numel() const
        {
            std::size_t result = 1;
            for(std::size_t dimension : shape_){
                result *=dimension;
            }
            return result;
        }
        std::size_t size() const
        {
            return numel();
        }
        std::vector<std::size_t> shape() const
        {
            return shape_;
        }
        T* data()
        {
            return data_.data();
        }
        Tensor(sycl::queue& queue, std::vector<std::size_t> shape): shape_(shape), data_(queue, numel())
        {

        }

    private:
        std::vector<std::size_t> shape_;
        DeviceBuffer<T> data_;
};

}