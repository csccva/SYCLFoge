from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gpt2"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
WEIGHTS = MODEL_DIR / "model.safetensors"
OUTPUT = ROOT / "build" / "reference_logits.bin"
INPUT_IDS_OUTPUT = ROOT / "build" / "reference_input_ids.txt"


def load_reference_model():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")

    config_path = MODEL_DIR / "config.json"

    if config_path.is_file():
        config = GPT2Config.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )
    else:
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            scale_attn_weights=True,
        )

    print(f"Loading local checkpoint: {WEIGHTS}", flush=True)

    model = GPT2LMHeadModel(config)

    state_dict = load_file(str(WEIGHTS), device="cpu")
    normalized_state_dict = {}

    for key, value in state_dict.items():
        # The local checkpoint uses keys such as h.0.ln_1.weight,
        # wte.weight, and ln_f.weight, without the "transformer." prefix.
        if not key.startswith(("transformer.", "lm_head.")):
            key = "transformer." + key

        # These are attention-mask buffers, not learned parameters.
        # Do not remove learned parameters such as attn.c_attn.bias.
        if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
            continue

        normalized_state_dict[key] = value

    incompatible = model.load_state_dict(
        normalized_state_dict,
        strict=False,
    )

    # GPT-2 ties the output projection to the token embedding matrix.
    model.tie_weights()

    missing = [
        key
        for key in incompatible.missing_keys
        if key != "lm_head.weight"
    ]
    unexpected = incompatible.unexpected_keys

    if missing or unexpected:
        print(f"Missing checkpoint keys: {missing}", file=sys.stderr)
        print(f"Unexpected checkpoint keys: {unexpected}", file=sys.stderr)
        raise RuntimeError(
            "Checkpoint does not match the GPT-2 reference architecture."
        )

    model.eval()
    model.to("cpu")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference next-token logits using CPU GPT-2."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Nice to meet you . My name is cris. What's your name?",
        help="Prompt used for numerical comparison",
    )
    args = parser.parse_args()

    torch.set_num_threads(4)

    tokenizer = GPT2TokenizerFast.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )

    model = load_reference_model()

    input_ids_list = tokenizer.encode(
        args.prompt,
        add_special_tokens=False,
    )

    if not input_ids_list:
        raise ValueError("The prompt produced no tokens.")

    if len(input_ids_list) > 1024:
        raise ValueError("Prompt exceeds GPT-2's 1024-token context length.")

    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long,
        device="cpu",
    )

    print(f"Prompt: {args.prompt}")
    print(f"Input token IDs: {input_ids_list}")

    with torch.inference_mode():
        output = model(input_ids=input_ids)
        logits = (
            output.logits[0, -1, :]
            .to(torch.float32)
            .cpu()
            .numpy()
            .copy()
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.asarray(logits, dtype=np.float32).tofile(OUTPUT)
    INPUT_IDS_OUTPUT.write_text(
        " ".join(map(str, input_ids_list)) + "\n",
        encoding="utf-8",
    )

    top_ids = np.argsort(logits)[-10:][::-1]

    print(f"Reference logits shape: {logits.shape}")
    print(
        f"Reference logits range: "
        f"[{logits.min():.6f}, {logits.max():.6f}]"
    )
    print("Top 10 next-token predictions:")

    for token_id in top_ids:
        print(
            f"  id={token_id:5d} "
            f"logit={logits[token_id]:12.6f} "
            f"text={tokenizer.decode([int(token_id)])!r}"
        )

    print(f"Saved reference logits: {OUTPUT}")
    print(f"Saved input token IDs: {INPUT_IDS_OUTPUT}")


if __name__ == "__main__":
    main()
