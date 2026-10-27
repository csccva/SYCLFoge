#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/linear.hpp"
#include "tinyml/transformer_block.hpp"

using T = float;

static constexpr T EPS = 1.0e-5f;
static constexpr T TOL = 5.0e-4f;

static constexpr std::size_t B = 2, L = 1024, D = 768, H = 12;
static constexpr std::size_t group_size = 64, tile_size = 4;
static constexpr std::size_t FF = 3072;

static void init_values(std::vector<T>& x, T scale, std::size_t period)
{
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<T>(static_cast<int>(i % period) -
                              static_cast<int>(period / 2)) * scale;
}

static void copy_from_device(sycl::queue& queue, tinyml::Tensor<T>& tensor, std::vector<T>& host)
{
    ops::copy_to_host(queue, host.data(), tensor).wait_and_throw();
}

static void set_linear(sycl::queue& queue, tinyml::Linear<T>& layer,
                       const std::vector<T>& W, const std::vector<T>& b)
{
    auto ew = ops::copy_to_device(queue, layer.weight(), W.data());
    auto eb = ops::copy_to_device(queue, layer.bias(), b.data(), {ew});
    eb.wait_and_throw();
}

static void layer_norm_cpu(const std::vector<T>& X, std::vector<T>& Y,
                           std::size_t rows, std::size_t Dm)
{
    for (std::size_t i = 0; i < rows; ++i) {
        T mean = 0.0f;
        for (std::size_t j = 0; j < Dm; ++j) mean += X[i * Dm + j];
        mean /= static_cast<T>(Dm);
        T var = 0.0f;
        for (std::size_t j = 0; j < Dm; ++j) {
            const T d = X[i * Dm + j] - mean;
            var += d * d;
        }
        var /= static_cast<T>(Dm);
        for (std::size_t j = 0; j < Dm; ++j)
            Y[i * Dm + j] = (X[i * Dm + j] - mean) / std::sqrt(var + EPS);
    }
}

static void linear_cpu(const std::vector<T>& X, const std::vector<T>& W,
                       const std::vector<T>& bias, std::vector<T>& Y,
                       std::size_t rows, std::size_t in_f, std::size_t out_f)
{
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < out_f; ++j) {
            T value = bias[j];
            for (std::size_t k = 0; k < in_f; ++k)
                value += X[i * in_f + k] * W[k * out_f + j];
            Y[i * out_f + j] = value;
        }
}

static void causal_attention_cpu(const std::vector<T>& Q, const std::vector<T>& K,
                                 const std::vector<T>& V, std::vector<T>& O,
                                 std::size_t Bn, std::size_t Ln, std::size_t Dm, std::size_t Hn)
{
    const std::size_t Dh = Dm / Hn;
    const T scale = std::sqrt(static_cast<T>(Dh));

    for (std::size_t b = 0; b < Bn; ++b)
        for (std::size_t h = 0; h < Hn; ++h) {
            const std::size_t start = h * Dh;
            for (std::size_t i = 0; i < Ln; ++i) {
                const std::size_t out_base = (b * Ln + i) * Dm + start;
                if (i == 0) {                       // softmax over one element is 1
                    for (std::size_t d = 0; d < Dh; ++d)
                        O[out_base + d] = V[b * Ln * Dm + start + d];
                    continue;
                }
                std::vector<T> scores(i + 1);
                T max_score = std::numeric_limits<T>::lowest();
                for (std::size_t j = 0; j <= i; ++j) {
                    T score = 0.0f;
                    for (std::size_t d = 0; d < Dh; ++d)
                        score += Q[(b * Ln + i) * Dm + start + d] *
                                 K[(b * Ln + j) * Dm + start + d];
                    scores[j] = score / scale;
                    max_score = std::max(max_score, scores[j]);
                }
                T denom = 0.0f;
                for (T& s : scores) {
                    s = std::exp(s - max_score);
                    denom += s;
                }
                for (std::size_t d = 0; d < Dh; ++d) {
                    T value = 0.0f;
                    for (std::size_t j = 0; j <= i; ++j)
                        value += (scores[j] / denom) * V[(b * Ln + j) * Dm + start + d];
                    O[(b * Ln + i) * Dm + start + d] = value;
                }
            }
        }
}

static bool compare(const char* name, const std::vector<T>& gpu, const std::vector<T>& cpu)
{
    T max_abs = 0.0f, max_rel = 0.0f;
    std::size_t max_index = 0, bad_index = gpu.size();
    bool finite = true;

    for (std::size_t i = 0; i < gpu.size(); ++i) {
        if (!std::isfinite(gpu[i])) {
            finite = false;
            if (bad_index == gpu.size()) bad_index = i;
            continue;
        }
        const T abs_err = std::abs(gpu[i] - cpu[i]);
        const T rel_err = abs_err / std::max(std::abs(cpu[i]), 1.0e-6f);
        if (abs_err > max_abs) { max_abs = abs_err; max_index = i; }
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > TOL && bad_index == gpu.size()) bad_index = i;
    }

    std::cout << name << ": max_abs=" << max_abs << " max_rel=" << max_rel;
    if (!finite) std::cout << " NON-FINITE at " << bad_index;
    else if (bad_index != gpu.size())
        std::cout << " FAIL at " << bad_index << " GPU=" << gpu[bad_index] << " CPU=" << cpu[bad_index];
    else std::cout << " PASS";
    std::cout << " (max index " << max_index << ")\n";
    return finite && bad_index == gpu.size();
}

// Shared by STEP 1 (standalone layers) and STEP 2 (layers owned by the block).
static bool run_chain(sycl::queue& queue, const char* label,
                      tinyml::Linear<T>& q_proj, tinyml::Linear<T>& k_proj,
                      tinyml::Linear<T>& v_proj, tinyml::Linear<T>& o_proj)
{
    std::cout << "\n" << label << '\n';

    const std::size_t rows = B * L, n = rows * D;

    tinyml::Tensor<T> X(queue, {B, L, D}), X_norm(queue, {B, L, D});
    tinyml::Tensor<T> Q(queue, {B, L, D}), K(queue, {B, L, D}), V(queue, {B, L, D});
    tinyml::Tensor<T> A(queue, {B, L, D}), R(queue, {B, L, D});
    tinyml::Tensor<T> gamma(queue, {D}), beta(queue, {D});

    std::vector<T> h_X(n), h_X_norm(n), h_Q(n), h_K(n), h_V(n), h_A(n), h_R(n);
    init_values(h_X, 0.03f, 17);

    std::vector<T> WQ(D * D), WK(D * D), WV(D * D), WO(D * D);
    std::vector<T> bQ(D), bK(D), bV(D), bO(D);
    init_values(WQ, 0.004f, 11);
    init_values(WK, 0.003f, 13);
    init_values(WV, 0.004f, 9);
    init_values(WO, 0.003f, 15);
    init_values(bQ, 0.002f, 7);
    init_values(bK, 0.002f, 9);
    init_values(bV, 0.002f, 11);
    init_values(bO, 0.002f, 5);

    std::vector<T> ones(D, 1.0f), zeros(D, 0.0f);

    auto eX = ops::copy_to_device(queue, X, h_X.data());
    auto eg = ops::copy_to_device(queue, gamma, ones.data());
    auto eb = ops::copy_to_device(queue, beta, zeros.data(), {eg});

    set_linear(queue, q_proj, WQ, bQ);
    set_linear(queue, k_proj, WK, bK);
    set_linear(queue, v_proj, WV, bV);
    set_linear(queue, o_proj, WO, bO);

    auto e_norm = ops::layer_norm_parallel(queue, X.data(), gamma.data(), beta.data(),
                                           X_norm.data(), rows, D, group_size, {eX, eb});
    auto eQ = q_proj.forward_3d(X_norm, Q, group_size, tile_size, {e_norm});
    auto eK = k_proj.forward_3d(X_norm, K, group_size, tile_size, {e_norm});
    auto eV = v_proj.forward_3d(X_norm, V, group_size, tile_size, {e_norm});
    auto eA = ops::multi_heads_attention_scores_online_strided_causal(
        queue, Q, K, V, A, H, tile_size, {eQ, eK, eV});
    auto eR = o_proj.forward_3d(A, R, group_size, tile_size, {eA});
    eR.wait_and_throw();

    copy_from_device(queue, X_norm, h_X_norm);
    copy_from_device(queue, Q, h_Q);
    copy_from_device(queue, K, h_K);
    copy_from_device(queue, V, h_V);
    copy_from_device(queue, A, h_A);
    copy_from_device(queue, R, h_R);

    std::vector<T> c_X_norm(n), c_Q(n), c_K(n), c_V(n), c_A(n), c_R(n);
    layer_norm_cpu(h_X, c_X_norm, rows, D);
    linear_cpu(c_X_norm, WQ, bQ, c_Q, rows, D, D);
    linear_cpu(c_X_norm, WK, bK, c_K, rows, D, D);
    linear_cpu(c_X_norm, WV, bV, c_V, rows, D, D);
    causal_attention_cpu(c_Q, c_K, c_V, c_A, B, L, D, H);
    linear_cpu(c_A, WO, bO, c_R, rows, D, D);

    // Invariant: token 0 attends only to itself, so A[0, :] == V[0, :].
    std::cout << "token-0 check (A must equal V):\n";
    for (std::size_t d = 0; d < 4; ++d)
        std::cout << "  d=" << d
                  << " gpuV=" << h_V[d] << " gpuA=" << h_A[d]
                  << " cpuV=" << c_V[d] << " cpuA=" << c_A[d] << '\n';

    bool ok = true;
    ok &= compare("X_norm", h_X_norm, c_X_norm);
    ok &= compare("Q", h_Q, c_Q);
    ok &= compare("K", h_K, c_K);
    ok &= compare("V", h_V, c_V);
    ok &= compare("A (attention)", h_A, c_A);
    ok &= compare("R (output projection)", h_R, c_R);
    std::cout << label << (ok ? " PASSED\n" : " FAILED\n");
    return ok;
}

// STEP 3: exercise the complete TransformerBlock::forward().
static bool run_block_forward(sycl::queue& queue)
{
    std::cout << "\nSTEP 3: TransformerBlock::forward()\n" << std::flush;

    const std::size_t rows = B * L;
    const std::size_t n = rows * D;
    const std::size_t n_ff = rows * FF;

    std::cerr << "[3.1] constructing block\n";
    tinyml::TransformerBlock<T> block(queue, D, H, FF);

    std::cerr << "[3.2] allocating tensors\n";
    tinyml::Tensor<T> X(queue, {B, L, D}), Y(queue, {B, L, D});
    tinyml::Tensor<T> X_norm(queue, {B, L, D});
    tinyml::Tensor<T> Q(queue, {B, L, D}), K(queue, {B, L, D}), V(queue, {B, L, D});
    tinyml::Tensor<T> A(queue, {B, L, D}), R(queue, {B, L, D});
    tinyml::Tensor<T> R_norm(queue, {B, L, D});
    tinyml::Tensor<T> F1(queue, {B, L, FF}), F1_gelu(queue, {B, L, FF});
    tinyml::Tensor<T> F2(queue, {B, L, D});

    std::vector<T> h_X(n), h_Y(n);
    init_values(h_X, 0.03f, 17);

    std::vector<T> WQ(D * D), WK(D * D), WV(D * D), WO(D * D);
    std::vector<T> bQ(D), bK(D), bV(D), bO(D);
    std::vector<T> W1(D * FF), W2(FF * D), b1(FF), b2(D);

    init_values(WQ, 0.004f, 11);
    init_values(WK, 0.003f, 13);
    init_values(WV, 0.004f, 9);
    init_values(WO, 0.003f, 15);
    init_values(W1, 0.003f, 13);
    init_values(W2, 0.003f, 11);
    init_values(bQ, 0.002f, 7);
    init_values(bK, 0.002f, 9);
    init_values(bV, 0.002f, 11);
    init_values(bO, 0.002f, 5);
    init_values(b1, 0.002f, 7);
    init_values(b2, 0.002f, 9);

    std::cerr << "[3.3] uploading inputs and weights\n";
    auto eX = ops::copy_to_device(queue, X, h_X.data());

    set_linear(queue, block.q_proj(), WQ, bQ);
    set_linear(queue, block.k_proj(), WK, bK);
    set_linear(queue, block.v_proj(), WV, bV);
    set_linear(queue, block.o_proj(), WO, bO);
    set_linear(queue, block.ff1(), W1, b1);
    set_linear(queue, block.ff2(), W2, b2);

    std::cerr << "[3.4] calling block.forward\n";
    auto e = block.forward(
        X, Y, X_norm, Q, K, V, A, R, R_norm,
        F1, F1_gelu, F2, group_size, tile_size, {eX});

    std::cerr << "[3.5] forward returned, waiting\n";
    e.wait_and_throw();
    std::cerr << "[3.6] forward finished\n";

    std::vector<T> g_X_norm(n), g_Q(n), g_K(n), g_V(n), g_A(n);
    std::vector<T> g_R(n), g_R_norm(n), g_F1(n_ff), g_F1_gelu(n_ff), g_F2(n);

    copy_from_device(queue, X_norm, g_X_norm);
    copy_from_device(queue, Q, g_Q);
    copy_from_device(queue, K, g_K);
    copy_from_device(queue, V, g_V);
    copy_from_device(queue, A, g_A);
    copy_from_device(queue, R, g_R);
    copy_from_device(queue, R_norm, g_R_norm);
    copy_from_device(queue, F1, g_F1);
    copy_from_device(queue, F1_gelu, g_F1_gelu);
    copy_from_device(queue, F2, g_F2);
    copy_from_device(queue, Y, h_Y);

    std::vector<T> c_X_norm(n), c_Q(n), c_K(n), c_V(n), c_A(n);
    std::vector<T> c_R(n), c_R_norm(n), c_F1(n_ff), c_F1_gelu(n_ff);
    std::vector<T> c_F2(n), c_Y(n);

    layer_norm_cpu(h_X, c_X_norm, rows, D);
    linear_cpu(c_X_norm, WQ, bQ, c_Q, rows, D, D);
    linear_cpu(c_X_norm, WK, bK, c_K, rows, D, D);
    linear_cpu(c_X_norm, WV, bV, c_V, rows, D, D);
    causal_attention_cpu(c_Q, c_K, c_V, c_A, B, L, D, H);
    linear_cpu(c_A, WO, bO, c_R, rows, D, D);

    for (std::size_t i = 0; i < n; ++i)
        c_Y[i] = h_X[i] + c_R[i];

    layer_norm_cpu(c_Y, c_R_norm, rows, D);
    linear_cpu(c_R_norm, W1, b1, c_F1, rows, D, FF);

    constexpr T gelu_c = 0.7978845608028654f;
    for (std::size_t i = 0; i < n_ff; ++i) {
        const T x = c_F1[i];
        c_F1_gelu[i] = 0.5f * x *
            (1.0f + std::tanh(gelu_c * (x + 0.044715f * x * x * x)));
    }

    linear_cpu(c_F1_gelu, W2, b2, c_F2, rows, FF, D);

    for (std::size_t i = 0; i < n; ++i)
        c_Y[i] += c_F2[i];

    bool ok = true;
    ok &= compare("X_norm", g_X_norm, c_X_norm);
    ok &= compare("Q", g_Q, c_Q);
    ok &= compare("K", g_K, c_K);
    ok &= compare("V", g_V, c_V);
    ok &= compare("A (attention)", g_A, c_A);
    ok &= compare("R (output projection)", g_R, c_R);
    ok &= compare("R_norm", g_R_norm, c_R_norm);
    ok &= compare("F1", g_F1, c_F1);
    ok &= compare("F1_gelu", g_F1_gelu, c_F1_gelu);
    ok &= compare("F2", g_F2, c_F2);
    ok &= compare("Y (final)", h_Y, c_Y);

    std::cout << (ok ? "STEP 3 PASSED\n" : "STEP 3 FAILED\n");
    return ok;
}

int main()
{
    try {
        sycl::queue queue{sycl::default_selector_v};
        std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';

        bool ok = true;

        {
            tinyml::Linear<T> q(queue, D, D), k(queue, D, D), v(queue, D, D), o(queue, D, D);
            ok &= run_chain(queue, "STEP 1: external components", q, k, v, o);
        }
        {
            tinyml::TransformerBlock<T> block(queue, D, H, FF);
            ok &= run_chain(queue, "STEP 2: TransformerBlock components",
                            block.q_proj(), block.k_proj(), block.v_proj(), block.o_proj());
        }
        {
            ok &= run_block_forward(queue);
        }

        std::cout << (ok ? "\nALL STEPS PASSED\n" : "\nSOME STEPS FAILED\n");
        return ok ? 0 : 1;
    }
    catch (const sycl::exception& e) { std::cerr << "SYCL exception: " << e.what() << '\n'; return 2; }
    catch (const std::exception& e)  { std::cerr << "Exception: " << e.what() << '\n'; return 3; }
}