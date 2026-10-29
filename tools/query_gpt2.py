
from pathlib import Path
import argparse
import subprocess
import sys

from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = ROOT / "build" / "gpt2_generate"

def main():
    parser = argparse.ArgumentParser(description="Query the SYCL GPT-2 Small implementation.")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"Build the C++ executable first: {EXECUTABLE}")

    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)

    if not input_ids:
        raise ValueError("The prompt produced no tokens.")

    if len(input_ids) + args.max_new_tokens > 1024:
        raise ValueError("Prompt tokens plus requested new tokens must not exceed 1024.")

    print(f"Prompt tokens: {len(input_ids)}", file=sys.stderr)

    command = [
    str(EXECUTABLE),
    "--token-ids",
    "--max-new-tokens",
    str(args.max_new_tokens),
    *[str(token_id) for token_id in input_ids],
]

    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    if result.returncode != 0:
        raise RuntimeError(f"SYCL inference failed with exit code {result.returncode}")

    generated_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("GENERATED_IDS:")),
        None,
    )

    if generated_line is None:
        raise RuntimeError(f"Could not find generated token IDs in output: {result.stdout}")

    generated_ids = [
        int(value) for value in generated_line.split(":", 1)[1].split()
    ]

    all_ids = input_ids + generated_ids

    print("\n--- Generated text ---")
    print(tokenizer.decode(all_ids, skip_special_tokens=True))
    print("\n--- Generated token IDs ---")
    print(generated_ids)

if __name__ == "__main__":
    main()
