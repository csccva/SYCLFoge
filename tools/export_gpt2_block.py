
from pathlib import Path
import argparse
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "models/gpt2/model.safetensors"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")
    if not 0 <= args.layer < 12:
        raise ValueError("GPT-2 Small has 12 Transformer blocks")

    state = load_file(str(WEIGHTS_PATH))

    if "transformer.h.0.ln_1.weight" in state:
        prefix = "transformer."
    elif "h.0.ln_1.weight" in state:
        prefix = ""
    else:
        raise KeyError("Could not identify GPT-2 Transformer weight names")

    base = f"{prefix}h.{args.layer}."
    output_dir = ROOT / "models/gpt2" / f"block{args.layer}"
    output_dir.mkdir(parents=True, exist_ok=True)

    def get(name):
        key = base + name
        if key not in state:
            raise KeyError(f"Missing checkpoint tensor: {key}")
        return np.asarray(state[key], dtype=np.float32)

    qkv_weight = get("attn.c_attn.weight")
    qkv_bias = get("attn.c_attn.bias")

    if qkv_weight.shape != (768, 2304) or qkv_bias.shape != (2304,):
        raise ValueError(f"Unexpected QKV shapes: {qkv_weight.shape}, {qkv_bias.shape}")

    q_weight, k_weight, v_weight = np.split(qkv_weight, 3, axis=1)
    q_bias, k_bias, v_bias = np.split(qkv_bias, 3)

    tensors = {
        "ln1_gamma": get("ln_1.weight"),
        "ln1_beta": get("ln_1.bias"),
        "q_weight": q_weight,
        "q_bias": q_bias,
        "k_weight": k_weight,
        "k_bias": k_bias,
        "v_weight": v_weight,
        "v_bias": v_bias,
        "attn_out_weight": get("attn.c_proj.weight"),
        "attn_out_bias": get("attn.c_proj.bias"),
        "ln2_gamma": get("ln_2.weight"),
        "ln2_beta": get("ln_2.bias"),
        "ff1_weight": get("mlp.c_fc.weight"),
        "ff1_bias": get("mlp.c_fc.bias"),
        "ff2_weight": get("mlp.c_proj.weight"),
        "ff2_bias": get("mlp.c_proj.bias"),
    }

    expected_shapes = {
        "ln1_gamma": (768,),
        "ln1_beta": (768,),
        "q_weight": (768, 768),
        "q_bias": (768,),
        "k_weight": (768, 768),
        "k_bias": (768,),
        "v_weight": (768, 768),
        "v_bias": (768,),
        "attn_out_weight": (768, 768),
        "attn_out_bias": (768,),
        "ln2_gamma": (768,),
        "ln2_beta": (768,),
        "ff1_weight": (768, 3072),
        "ff1_bias": (3072,),
        "ff2_weight": (3072, 768),
        "ff2_bias": (768,),
    }

    for name, array in tensors.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name}: expected {expected_shapes[name]}, got {array.shape}")
        path = output_dir / f"{name}.bin"
        np.ascontiguousarray(array, dtype=np.float32).tofile(path)
        print(f"{path.relative_to(ROOT)}: shape={array.shape}, bytes={path.stat().st_size}")

    print(f"\nExported GPT-2 block {args.layer} to {output_dir}")

if __name__ == "__main__":
    main()
