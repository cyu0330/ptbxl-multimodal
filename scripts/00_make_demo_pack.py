# scripts/00_make_demo_pack.py
#
# Build a tiny, shareable demo pack (single-modal + multimodal) from PTB-XL.
# Output:
#   data/demo/single/sample_*.npz      (ecg, y)
#   data/demo/multimodal/sample_*.npz  (ecg, demo, y)
#   data/demo/meta.csv                (file, modality, y_true, chosen_for, etc.)

import os
import argparse
import numpy as np
import pandas as pd

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset

CLASSES = ["MI", "STTC", "HYP", "CD", "NORM"]


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _y_to_str(y: np.ndarray) -> str:
    # e.g. "MI=1;STTC=0;..."
    parts = [f"{CLASSES[i]}={int(y[i])}" for i in range(len(CLASSES))]
    return ";".join(parts)


def _is_all_zero(y: np.ndarray) -> bool:
    return int(np.sum(y)) == 0


def _pick_indices_covering_classes(ds, max_per_class=1, extra_all_zero=1, seed=42):
    """
    Pick indices to cover each class (positive samples), plus some all-zero samples.
    This is heuristic and deterministic given the seed.
    """
    rng = np.random.default_rng(seed)

    # Collect indices per class where y[c]==1
    per_class = {c: [] for c in range(len(CLASSES))}
    all_zero = []

    for i in range(len(ds)):
        item = ds[i]
        # PTBXLDataset: (x, y) ; PTBXLECGMultimodalDataset: (x_ecg, x_demo, y)
        y = item[-1].numpy()
        if _is_all_zero(y):
            all_zero.append(i)
        for c in range(len(CLASSES)):
            if int(y[c]) == 1:
                per_class[c].append(i)

    chosen = []
    chosen_for = {}

    # Pick positives per class
    for c in range(len(CLASSES)):
        pool = per_class[c]
        if len(pool) == 0:
            continue
        rng.shuffle(pool)
        take = pool[:max_per_class]
        for idx in take:
            if idx not in chosen:
                chosen.append(idx)
                chosen_for[idx] = f"pos_{CLASSES[c]}"

    # Pick extra all-zero (normal-ish)
    if len(all_zero) > 0 and extra_all_zero > 0:
        rng.shuffle(all_zero)
        for idx in all_zero[:extra_all_zero]:
            if idx not in chosen:
                chosen.append(idx)
                chosen_for[idx] = "all_zero"

    return chosen, chosen_for


def export_single(ds_single, out_dir, indices, chosen_for, meta_rows, prefix="single"):
    _safe_mkdir(out_dir)

    for k, idx in enumerate(indices):
        x, y = ds_single[idx]
        ecg = x.numpy().astype(np.float32)
        y_np = y.numpy().astype(np.float32)

        fname = f"{prefix}_sample_{k:02d}.npz"
        fpath = os.path.join(out_dir, fname)
        np.savez_compressed(fpath, ecg=ecg, y=y_np, classes=np.array(CLASSES))

        meta_rows.append({
            "file": f"single/{fname}",
            "modality": "single",
            "index_in_split": int(idx),
            "chosen_for": chosen_for.get(idx, "unknown"),
            "y_true": _y_to_str(y_np),
            "y_sum": int(np.sum(y_np)),
            "ecg_shape": str(tuple(ecg.shape)),
        })


def export_multimodal(ds_mm, out_dir, indices, chosen_for, meta_rows, prefix="mm"):
    _safe_mkdir(out_dir)

    for k, idx in enumerate(indices):
        x_ecg, x_demo, y = ds_mm[idx]
        ecg = x_ecg.numpy().astype(np.float32)
        demo = x_demo.numpy().astype(np.float32)
        y_np = y.numpy().astype(np.float32)

        fname = f"{prefix}_sample_{k:02d}.npz"
        fpath = os.path.join(out_dir, fname)
        np.savez_compressed(fpath, ecg=ecg, demo=demo, y=y_np, classes=np.array(CLASSES))

        meta_rows.append({
            "file": f"multimodal/{fname}",
            "modality": "multimodal",
            "index_in_split": int(idx),
            "chosen_for": chosen_for.get(idx, "unknown"),
            "y_true": _y_to_str(y_np),
            "y_sum": int(np.sum(y_np)),
            "ecg_shape": str(tuple(ecg.shape)),
            "demo_shape": str(tuple(demo.shape)),
        })


def main(args):
    set_seed(args.seed)

    out_root = args.out_root
    single_dir = os.path.join(out_root, "single")
    mm_dir = os.path.join(out_root, "multimodal")
    _safe_mkdir(single_dir)
    _safe_mkdir(mm_dir)

    # Load datasets (test split)
    ds_single = PTBXLDataset(
        base_dir=args.base_dir,
        split="test",
        classes=CLASSES,
        normalize=args.normalize,
    )
    ds_mm = PTBXLECGMultimodalDataset(
        base_dir=args.base_dir,
        split="test",
        classes=CLASSES,
        normalize=args.normalize,
    )
    print(f"[INFO] PTBXLDataset(test) size = {len(ds_single)}")
    print(f"[INFO] PTBXLECGMultimodalDataset(test) size = {len(ds_mm)}")

    # Choose indices to cover classes
    idx_single, chosen_for_single = _pick_indices_covering_classes(
        ds_single,
        max_per_class=args.per_class,
        extra_all_zero=args.extra_all_zero,
        seed=args.seed,
    )
    idx_mm, chosen_for_mm = _pick_indices_covering_classes(
        ds_mm,
        max_per_class=args.per_class,
        extra_all_zero=args.extra_all_zero,
        seed=args.seed,
    )

    print(f"[INFO] Chosen single indices: {idx_single}")
    print(f"[INFO] Chosen multimodal indices: {idx_mm}")

    meta_rows = []
    export_single(ds_single, single_dir, idx_single, chosen_for_single, meta_rows, prefix="single")
    export_multimodal(ds_mm, mm_dir, idx_mm, chosen_for_mm, meta_rows, prefix="mm")

    meta_path = os.path.join(out_root, "meta.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"[SAVE] meta.csv -> {meta_path}")
    print("[DONE] Demo pack created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="PTB-XL base directory.")
    parser.add_argument("--out_root", type=str, default="data/demo", help="Output root directory.")
    parser.add_argument("--normalize", type=str, default="per_lead", help="Normalization mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--per_class", type=int, default=1, help="How many positive samples per class.")
    parser.add_argument("--extra_all_zero", type=int, default=2, help="Extra all-zero (normal-ish) samples.")
    args = parser.parse_args()
    main(args)
