from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "build" / "reference_logits.bin"
SYCL = ROOT / "build" / "sycl_logits.bin"
VOCAB_SIZE = 50257


def load_logits(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    values = np.fromfile(path, dtype=np.float32)

    if values.size != VOCAB_SIZE:
        raise ValueError(
            f"{path} contains {values.size} values; "
            f"expected {VOCAB_SIZE}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path}")

    return values.astype(np.float64)


def main():
    ref = load_logits(REFERENCE)
    sycl = load_logits(SYCL)

    diff = sycl - ref
    abs_diff = np.abs(diff)

    ref_centered = ref - ref.max()
    sycl_centered = sycl - sycl.max()
    centered_diff = sycl_centered - ref_centered

    rmse = np.sqrt(np.mean(diff ** 2))
    centered_rmse = np.sqrt(np.mean(centered_diff ** 2))

    cosine = np.dot(ref, sycl) / (
        np.linalg.norm(ref) * np.linalg.norm(sycl)
    )

    correlation = np.corrcoef(ref, sycl)[0, 1]

    ref_top1 = int(np.argmax(ref))
    sycl_top1 = int(np.argmax(sycl))

    ref_top10 = set(np.argsort(ref)[-10:])
    sycl_top10 = set(np.argsort(sycl)[-10:])

    print("=== GPT-2 SYCL vs reference logits ===")
    print(f"Vocabulary size:             {VOCAB_SIZE}")
    print(f"Reference range:              [{ref.min():.8f}, {ref.max():.8f}]")
    print(f"SYCL range:                   [{sycl.min():.8f}, {sycl.max():.8f}]")
    print()
    print(f"Maximum absolute error:       {abs_diff.max():.10g}")
    print(f"Mean absolute error:          {abs_diff.mean():.10g}")
    print(f"RMSE:                         {rmse:.10g}")
    print(f"Mean signed error (SYCL-ref): {diff.mean():.10g}")
    print(f"Maximum centered error:       {np.max(np.abs(centered_diff)):.10g}")
    print(f"Centered RMSE:                {centered_rmse:.10g}")
    print(f"Cosine similarity:            {cosine:.12f}")
    print(f"Pearson correlation:           {correlation:.12f}")
    print()
    print(f"Reference top token:          {ref_top1}")
    print(f"SYCL top token:               {sycl_top1}")
    print(f"Top-1 prediction matches:     {ref_top1 == sycl_top1}")
    print(f"Top-10 overlap:               {len(ref_top10 & sycl_top10)}/10")
    print()
    print("Largest absolute logit differences:")
    for i in np.argsort(abs_diff)[-10:][::-1]:
        print(
            f"  ID {i:5d}: ref={ref[i]: .8f}, "
            f"SYCL={sycl[i]: .8f}, diff={diff[i]: .3e}"
        )


if __name__ == "__main__":
    main()
