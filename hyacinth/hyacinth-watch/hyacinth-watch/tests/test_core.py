import numpy as np
import pandas as pd
from affine import Affine

from hyacinthwatch.indices import fai, ndvi, compute_indices
from hyacinthwatch.weak_labels import valid_mask, permanent_water, weak_hyacinth_mask
from hyacinthwatch.chips import make_chips, spatial_split
from hyacinthwatch.timeseries import scene_stats, monthly


def test_fai_zero_on_flat_spectrum_and_positive_on_nir_bump():
    assert abs(fai(0.05, 0.05, 0.05)) < 1e-9
    assert abs(fai(0.05, 0.15, 0.05) - 0.10) < 1e-9


def test_ndvi_range():
    assert abs(ndvi(0.3, 0.1) - 0.5) < 1e-4


def test_compute_indices_keys():
    b = {k: np.full((2, 2), 0.1) for k in ("B03", "B04", "B08", "B11")}
    assert set(compute_indices(b)) == {"ndvi", "ndwi", "mndwi", "fai"}


def test_valid_mask_flags_clouds():
    scl = np.array([[4, 8], [9, 6]])
    assert valid_mask(scl).tolist() == [[True, False], [False, True]]


def test_permanent_water_and_weak_mask():
    T, H, W = 4, 6, 6
    m = np.full((T, H, W), -0.5)
    m[:, :3] = 0.3
    v = np.ones((T, H, W), bool)
    water = permanent_water(m, v)
    assert water[:3].all() and not water[3:].any()
    nd = np.full((H, W), 0.4)
    fa = np.full((H, W), 0.05)
    wk = weak_hyacinth_mask(nd, fa, water, np.ones((H, W), bool))
    assert wk[:3].all() and not wk[3:].any()


def test_chips_and_spatial_split_no_location_leakage(tmp_path):
    H = W = 512
    img = np.random.rand(5, H, W).astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[:100, :100] = 1
    ones = np.ones((H, W), bool)
    frames = []
    for sid, x0 in (("a", 0.0), ("b", 0.0)):  # two dates, same footprint
        t = Affine(10, 0, x0, 0, -10, 0.0)
        frames.append(make_chips(img, mask, ones, ones, t, sid, tmp_path, size=128, stride=128))
    idx = pd.concat(frames, ignore_index=True)
    assert len(idx) == 32
    out = spatial_split(idx, block_m=1280, val_frac=0.25, test_frac=0.25)
    locs = out.groupby(["x", "y"])["split"].nunique()
    assert (locs == 1).all()  # same place never appears in two splits
    assert set(out.split) <= {"train", "val", "test"}


def test_timeseries_area_and_monthly():
    prob = np.zeros((10, 10)); prob[:2] = 0.9
    water = np.ones((10, 10), bool); valid = np.ones((10, 10), bool)
    s = scene_stats("2024-01-05", prob, water, valid)
    assert abs(s["hyacinth_ha"] - 20 * 0.01) < 1e-9 and abs(s["cover_pct"] - 20) < 1e-9
    df = pd.DataFrame([s, scene_stats("2024-01-20", prob, water, valid)])
    assert len(monthly(df)) == 1
