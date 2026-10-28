"""Chip generation and leakage-safe spatial splitting."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_chips(image, mask, valid, water, transform, scene_id, out_dir,
               size=256, stride=256, min_valid=0.9, min_water=0.2):
    """Cut (C, H, W) image + (H, W) mask into .npz chips; return an index DataFrame.

    Chips are kept only if enough pixels are valid (cloud-free) and enough are
    water. The index stores projected x/y of each chip's upper-left corner,
    which `spatial_split` uses to keep locations out of more than one split.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, H, W = image.shape
    rows = []
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            v = valid[r:r + size, c:c + size]
            w = water[r:r + size, c:c + size]
            if v.mean() < min_valid or w.mean() < min_water:
                continue
            m = mask[r:r + size, c:c + size]
            img = np.nan_to_num(image[:, r:r + size, c:c + size]).astype(np.float32)
            path = out / f"{scene_id}_{r}_{c}.npz"
            np.savez_compressed(path, image=img, mask=m.astype(np.uint8),
                                valid=v.astype(np.uint8))
            rows.append({
                "scene_id": scene_id, "row": r, "col": c,
                "x": transform.c + c * transform.a,
                "y": transform.f + r * transform.e,
                "path": str(path),
                "valid_frac": float(v.mean()),
                "pos_frac": float((m == 1).mean()),
            })
    return pd.DataFrame(rows)


def spatial_split(index, block_m=5000.0, val_frac=0.15, test_frac=0.15, seed=42):
    """Assign train/val/test by geographic block, across ALL dates.

    Random chip splits leak: the same patch of lake on neighbouring dates looks
    almost identical, inflating scores. Splitting by location avoids that.
    """
    gx = np.floor(index["x"] / block_m).astype(int)
    gy = np.floor(index["y"] / block_m).astype(int)
    group = gx.astype(str) + "_" + gy.astype(str)
    uniq = np.array(sorted(group.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test = set(uniq[:n_test])
    val = set(uniq[n_test:n_test + n_val])
    out = index.copy()
    out["split"] = np.where(group.isin(test), "test",
                            np.where(group.isin(val), "val", "train"))
    return out
