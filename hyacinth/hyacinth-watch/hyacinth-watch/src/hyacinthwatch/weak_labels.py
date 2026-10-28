"""Spectral weak labels and mask helpers.

Weak labels are NOISY. Floating cyanobacterial blooms also light up FAI, so
they must be refined (SAM-assisted or manual, QGIS) before being trusted as
ground truth, and the test split should use hand-corrected labels only.
"""
import numpy as np

# SCL: 0 nodata, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)


def valid_mask(scl):
    return ~np.isin(np.asarray(scl), INVALID_SCL)


def permanent_water(mndwi_stack, valid_stack, min_freq=0.5):
    """Image-derived fallback water mask from a (T, H, W) stack.

    Limitation: chronically infested areas have MNDWI < 0 and can drop out.
    Prefer a fixed lake polygon or a JRC surface-water occurrence layer.
    """
    wet = (np.asarray(mndwi_stack) > 0) & np.asarray(valid_stack)
    obs = np.asarray(valid_stack).sum(0)
    freq = wet.sum(0) / np.maximum(obs, 1)
    return (freq >= min_freq) & (obs > 0)


def polygon_mask(path, like):
    """Rasterize a polygon file onto the grid of DataArray `like` (band, y, x)."""
    import geopandas as gpd
    from rasterio import features

    ref = like.isel(band=0) if "band" in like.dims else like
    gdf = gpd.read_file(path).to_crs(ref.rio.crs)
    return features.rasterize(
        ((g, 1) for g in gdf.geometry),
        out_shape=ref.shape,
        transform=ref.rio.transform(),
        fill=0,
        dtype="uint8",
    ).astype(bool)


def weak_hyacinth_mask(ndvi_arr, fai_arr, water, valid, ndvi_min=0.25, fai_min=0.02):
    m = water & valid & (ndvi_arr >= ndvi_min) & (fai_arr >= fai_min)
    return m.astype(np.uint8)


def load_refined_mask(path, like):
    """Read a hand-refined label raster (1 = hyacinth) onto the grid of `like`."""
    import rioxarray as rxr

    ref = like.isel(band=0) if "band" in like.dims else like
    da = rxr.open_rasterio(path).squeeze("band", drop=True)
    da = da.rio.reproject_match(ref)
    return (da.values == 1).astype(np.uint8)
