# .venv/bin/python tools/export_gpt2_embeddings.py
from pathlib import Path
import numpy as np
from safetensors import safe_open

root = Path(__file__).resolve().parents[1]
checkpoint = root / "models/gpt2/model.safetensors"
output_dir = root / "models/gpt2"

with safe_open(checkpoint, framework="np") as f:
    tensors = [
        ("wte.weight", "token_embeddings.bin", (50257, 768)),
        ("wpe.weight", "position_embeddings.bin", (1024, 768)),
    ]

    for name, filename, expected_shape in tensors:
        tensor = f.get_tensor(name)

        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name}: expected {expected_shape}, got {tensor.shape}"
            )
        if tensor.dtype != np.float32:
            raise TypeError(
                f"{name}: expected float32, got {tensor.dtype}"
            )

        tensor = np.ascontiguousarray(tensor, dtype="<f4")
        path = output_dir / filename
        tensor.tofile(path)
        print(f"{filename}: shape={tensor.shape}, bytes={path.stat().st_size}")

print("Embedding export completed.")
