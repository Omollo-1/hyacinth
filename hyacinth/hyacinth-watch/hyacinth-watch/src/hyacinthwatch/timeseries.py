"""Area and cover statistics per scene, and monthly aggregation."""
import numpy as np
import pandas as pd


def scene_stats(date, prob, water, valid, thr=0.5, pixel_m=10.0):
    px_ha = pixel_m ** 2 / 10_000.0
    hy = (prob >= thr) & water & valid
    obs = water & valid
    return {
        "date": pd.Timestamp(date),
        "hyacinth_ha": float(hy.sum() * px_ha),
        "observed_water_ha": float(obs.sum() * px_ha),
        "observed_frac": float(obs.sum() / max(water.sum(), 1)),
        "cover_pct": float(100.0 * hy.sum() / max(obs.sum(), 1)),
    }


def monthly(df, min_observed_frac=0.5):
    """Median per month, ignoring scenes where too little water was visible."""
    d = df[df.observed_frac >= min_observed_frac].copy()
    d = d.set_index("date").sort_index()
    return d[["hyacinth_ha", "cover_pct"]].resample("MS").median()
