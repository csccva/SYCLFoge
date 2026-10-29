
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = 768
N_HEADS = 12
D_HEAD = D // N_HEADS
D_FF = 3072
EPSILON = 1.0e-5

def load_weight(directory, name, shape):
    path = directory / f"{name}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing weight: {path}")
    values = np.fromfile(path, dtype=np.float32)
    expected_size = int(np.prod(shape))
    if values.size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} values, got {values.size}")
    return values.reshape(shape).astype(np.float64)

def layer_norm(x, gamma, beta):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + EPSILON) * gamma + beta

def linear(x, weight, bias):
    return x @ weight + bias

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    args = parser.parse_args()

    if not 0 <= args.layer < 12:
        raise ValueError("GPT-2 Small has 12 Transformer blocks")
    if not 1 <= args.sequence_length <= 1024:
        raise ValueError("Sequence length must be between 1 and 1024")

    L = args.sequence_length
    directory = ROOT / "models" / "gpt2" / f"block{args.layer}"

    # Deterministic float32 input shared with the C++ test.
    indices = np.arange(L * D)
    x32 = (((indices % 37) - 18).astype(np.float32) * np.float32(0.01)).reshape(1, L, D)
    x = x32.astype(np.float64)
    x32.tofile(directory / "test_input.bin")

    ln1_gamma = load_weight(directory, "ln1_gamma", (D,))
    ln1_beta = load_weight(directory, "ln1_beta", (D,))
    q_weight = load_weight(directory, "q_weight", (D, D))
    q_bias = load_weight(directory, "q_bias", (D,))
    k_weight = load_weight(directory, "k_weight", (D, D))
    k_bias = load_weight(directory, "k_bias", (D,))
    v_weight = load_weight(directory, "v_weight", (D, D))
    v_bias = load_weight(directory, "v_bias", (D,))
    out_weight = load_weight(directory, "attn_out_weight", (D, D))
    out_bias = load_weight(directory, "attn_out_bias", (D,))
    ln2_gamma = load_weight(directory, "ln2_gamma", (D,))
    ln2_beta = load_weight(directory, "ln2_beta", (D,))
    ff1_weight = load_weight(directory, "ff1_weight", (D, D_FF))
    ff1_bias = load_weight(directory, "ff1_bias", (D_FF,))
    ff2_weight = load_weight(directory, "ff2_weight", (D_FF, D))
    ff2_bias = load_weight(directory, "ff2_bias", (D,))

    # Pre-LayerNorm and QKV projections.
    x_norm = layer_norm(x, ln1_gamma, ln1_beta)
    q = linear(x_norm, q_weight, q_bias).reshape(1, L, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    k = linear(x_norm, k_weight, k_bias).reshape(1, L, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    v = linear(x_norm, v_weight, v_bias).reshape(1, L, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D_HEAD)
    scores = np.where(np.triu(np.ones((L, L), dtype=bool), k=1)[None, None, :, :], -1.0e30, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

    context = (probabilities @ v).transpose(0, 2, 1, 3).reshape(1, L, D)
    attention_output = linear(context, out_weight, out_bias)
    residual = x + attention_output

    # Second pre-LayerNorm and GPT-2's GELU-new activation.
    x_norm = layer_norm(residual, ln2_gamma, ln2_beta)
    hidden = linear(x_norm, ff1_weight, ff1_bias)
    hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
    output = residual + linear(hidden, ff2_weight, ff2_bias)

    output32 = np.asarray(output, dtype=np.float32)
    output32.tofile(directory / "reference_output.bin")
    print(f"Input shape: {x32.shape}")
    print(f"Reference output shape: {output32.shape}")
    print(f"Output range: [{output32.min():.7g}, {output32.max():.7g}]")
    print(f"Saved: {directory / 'test_input.bin'}")
    print(f"Saved: {directory / 'reference_output.bin'}")

if __name__ == "__main__":
    main()
