#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>


#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/linear.hpp"
#include "tinyml/ops.hpp"

namespace tinyml {

    template <typename T>
    class TransformerBlock
    {
        public:
        TransformerBlock(sycl::queue& queue, std::size_t d_model, std::size_t nheads, std::size_t d_ff) : queue_(queue), d_model_(d_model), nheads_(nheads), d_ff_(d_ff), Q_proj_(queue, d_model, d_model), K_proj_(queue, d_model, d_model), V_proj_(queue, d_model, d_model), O_proj_(queue, d_model, d_model), FF1_(queue, d_model, d_ff), FF2_(queue, d_ff, d_model), gamma1_(queue, {d_model}), beta1_(queue, {d_model}), gamma2_(queue, {d_model}), beta2_(queue, {d_model})
        {
            if (d_model == 0 || nheads == 0 || d_ff == 0 || d_model % nheads != 0) {
                throw std::invalid_argument("Invalid TransformerBlock dimensions");
            }
            auto e1 = ops::fill(queue_, gamma1_, T{1}, 256);
            auto e2 = ops::fill(queue_, beta1_, T{0}, 256, {e1});
            auto e3 = ops::fill(queue_, gamma2_, T{1}, 256, {e2});
            auto e4 = ops::fill(queue_, beta2_, T{0}, 256, {e3});
            e4.wait();
        }
        Linear<T>& q_proj() { return Q_proj_; }
        Linear<T>& k_proj() { return K_proj_; }
        Linear<T>& v_proj() { return V_proj_; }
        Linear<T>& o_proj() { return O_proj_; }
        Linear<T>& ff1() { return FF1_; }
        Linear<T>& ff2() { return FF2_; }

        Tensor<T>& ln1_gamma() { return gamma1_; }
        Tensor<T>& ln1_beta()  { return beta1_; }
        Tensor<T>& ln2_gamma() { return gamma2_; }
        Tensor<T>& ln2_beta()  { return beta2_; }

        sycl::event forward(Tensor<T>& X,Tensor<T>& Y,Tensor<T>& X_norm,Tensor<T>& Q,Tensor<T>& K,Tensor<T>& V,Tensor<T>& A,Tensor<T>& R,Tensor<T>& R_norm,Tensor<T>& F1,Tensor<T>& F1_gelu,Tensor<T>& F2,std::size_t group_size,std::size_t tile_size,const std::vector<sycl::event>& dependencies = {})
        {
            auto e1 = ops::layer_norm_parallel(queue_,X.data(),gamma1_.data(),beta1_.data(),X_norm.data(),X.size()/d_model_,d_model_,group_size,dependencies);
            auto e2 = Q_proj_.forward_3d(X_norm,Q,group_size,tile_size,{e1});
            auto e3 = K_proj_.forward_3d(X_norm,K,group_size,tile_size,{e1});
            auto e4 = V_proj_.forward_3d(X_norm,V,group_size,tile_size,{e1});
            auto e5 = ops::multi_heads_attention_scores_online_strided_causal(queue_,Q,K,V,A,nheads_,tile_size,{e2,e3,e4});
            e5.wait_and_throw();
            auto e6_matmul = ops::matmul_tiled(queue_,A.data(),O_proj_.weight().data(),R.data(),X.size()/d_model_,d_model_,d_model_,tile_size,{e5});
            auto e6 = ops::add_bias(queue_,R.data(),O_proj_.bias().data(),R.data(),X.size()/d_model_,d_model_,group_size,{e6_matmul});
            auto e7 = ops::add(queue_,X,R,Y,group_size,{e6});
            auto e8 = ops::layer_norm_parallel(queue_,Y.data(),gamma2_.data(),beta2_.data(),R_norm.data(),X.size()/d_model_,d_model_,group_size,{e7});
            auto e9 = FF1_.forward_3d(R_norm,F1,group_size,tile_size,{e8});
            auto e10 = ops::gelu_approx(queue_,F1,F1_gelu,group_size,{e9});
            auto e11 = FF2_.forward_3d(F1_gelu,F2,group_size,tile_size,{e10});
            auto e12 = ops::add(queue_,Y,F2,Y,group_size,{e11});
            return e12;
        }
        private:
            sycl::queue& queue_;
            std::size_t d_model_;
            std::size_t nheads_;
            std::size_t d_ff_;
            Linear<T> Q_proj_;
            Linear<T> K_proj_;
            Linear<T> V_proj_;
            Linear<T> O_proj_;
            Linear<T> FF1_;
            Linear<T> FF2_;
            Tensor<T> gamma1_;
            Tensor<T> beta1_;
            Tensor<T> gamma2_;
            Tensor<T> beta2_;
    };
}
