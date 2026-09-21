#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>

namespace tinyml {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer(sycl::queue& queue, std::size_t count): queue_(queue), count_(count)
    {
        data_ = sycl::malloc_device<T>(count_, queue_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        sycl::free(data_, queue_);
    }

    T* data()
    {
        return data_;
    }

    std::size_t size() const
    {
        return count_;
    }

private:
    sycl::queue& queue_;
    T* data_;
    std::size_t count_;
};

} // namespace tinyml