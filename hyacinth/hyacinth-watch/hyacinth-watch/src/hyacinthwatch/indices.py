"""Spectral indices. Work on numpy arrays or xarray DataArrays (reflectance 0-1)."""
EPS = 1e-6


def _nd(a, b):
    return (a - b) / (a + b + EPS)


def ndvi(nir, red):
    return _nd(nir, red)


def ndwi(green, nir):
    """McFeeters NDWI."""
    return _nd(green, nir)


def mndwi(green, swir):
    """Xu's modified NDWI."""
    return _nd(green, swir)


def fai(red, nir, swir, l_red=665.0, l_nir=842.0, l_swir=1610.0):
    """Floating Algae Index (Hu, 2009): NIR minus a red-SWIR baseline.

    Defaults are Sentinel-2 band centres (B04, B08, B11) in nm.
    """
    baseline = red + (swir - red) * (l_nir - l_red) / (l_swir - l_red)
    return nir - baseline


def compute_indices(b):
    """b: dict with keys B03, B04, B08, B11 -> dict of index arrays."""
    return {
        "ndvi": ndvi(b["B08"], b["B04"]),
        "ndwi": ndwi(b["B03"], b["B08"]),
        "mndwi": mndwi(b["B03"], b["B11"]),
        "fai": fai(b["B04"], b["B08"], b["B11"]),
    }


def bands_dict(image, bands):
    """Split a (C, H, W) array into {band_name: (H, W)}."""
    return {name: image[i] for i, name in enumerate(bands)}
