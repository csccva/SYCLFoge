
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/gpt2/model.safetensors"
OUTPUT = ROOT / "models/gpt2"

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    state = load_file(str(CHECKPOINT))

    if "transformer.h.0.ln_1.weight" in state:
        prefix = "transformer."
    elif "h.0.ln_1.weight" in state:
        prefix = ""
    else:
        raise KeyError("Cannot determine GPT-2 checkpoint naming convention")

    parameters = {
        "final_ln_gamma.bin": prefix + "ln_f.weight",
        "final_ln_beta.bin": prefix + "ln_f.bias",
    }

    for filename, key in parameters.items():
        if key not in state:
            raise KeyError(f"Missing checkpoint tensor: {key}")

        values = np.asarray(state[key], dtype=np.float32)

        if values.shape != (768,):
            raise ValueError(f"{key}: expected shape (768,), got {values.shape}")

        path = OUTPUT / filename
        np.ascontiguousarray(values).tofile(path)
        print(f"{path.relative_to(ROOT)}: shape={values.shape}, bytes={path.stat().st_size}")

    print("GPT-2 final LayerNorm export passed.")

if __name__ == "__main__":
    main()
