#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include "tinyml/tensor.hpp"
#include "tinyml/device_buffer.hpp"

namespace tinyml {

    template <typename T>
    class Linear {
        public:
        Linear(sycl::queue& queue, std::size_t in_features, std::size_t out_features): queue_(queue), W_(queue, {in_features,out_features}), b_(queue,{out_features})
        {
            
        }
        Tensor<T>& weight()
        {
            return W_;
        }
        Tensor<T>& bias()
        {
            return b_;
        }
        sycl::event forward(Tensor<T>& X,Tensor<T>& Y, std::size_t group_size,std::size_t tile_size, const std::vector<sycl::event>& dependencies)
        {
            
            if (X.shape().size() != 2 || Y.shape().size() != 2 || X.shape()[1] != W_.shape()[0] || Y.shape()[1] != W_.shape()[1] || Y.shape()[0] != X.shape()[0]){
                throw std::invalid_argument("Linear: incompatible tensor shapes");
            }
            sycl::event e_matmul=ops::matmul_tiled(queue_,X, W_,Y,tile_size,dependencies );
            return ops::add_bias(queue_,Y,b_,Y,group_size,{e_matmul});            
        }
        private:
            sycl::queue& queue_;
            Tensor<T> W_; // weights   
            Tensor<T> b_; //biases
    };  
}