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
        sycl::event forward_3d(Tensor<T>& X,Tensor<T>& Y,std::size_t group_size,std::size_t tile_size,const std::vector<sycl::event>& dependencies)
        {
            if(X.shape().size()!=3 || Y.shape().size()!=3 || X.shape()[2] != W_.shape()[0] || Y.shape()[2] != W_.shape()[1] || Y.shape()[0] != X.shape()[0] || Y.shape()[1] != X.shape()[1]){
                throw std::invalid_argument("Linear::forward_3d: tensors must be 3D");
            }
            std::size_t B=X.shape()[0];
            std::size_t L=X.shape()[1];
            std::size_t M=B*L;
            std::size_t K=X.shape()[2];
            std::size_t N=Y.shape()[2];
            sycl::event e_matmul=ops::matmul_tiled(queue_,X.data(), W_.data(), Y.data(),M, K, N,tile_size, dependencies);
            return ops::add_bias(queue_,Y.data(),b_.data(),Y.data(), M,N, group_size,{e_matmul});  
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