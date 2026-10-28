"""Sentinel-2 L2A access through the Microsoft Planetary Computer STAC API."""
import numpy as np
import planetary_computer as pc
import rioxarray as rxr
import xarray as xr
from pystac_client import Client
from rasterio.enums import Resampling

BANDS = ["B02", "B03", "B04", "B08", "B11"]


def search_scenes(bbox, date_range, max_cloud=30,
                  collection="sentinel-2-l2a",
                  stac_url="https://planetarycomputer.microsoft.com/api/stac/v1"):
    catalog = Client.open(stac_url, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return sorted(search.items(), key=lambda i: i.datetime)


def _boa_offset(item):
    """Processing baseline >= 04.00 (from 2022-01-25) adds a +1000 DN offset."""
    baseline = float(item.properties.get("s2:processing_baseline", "0"))
    return 1000.0 if baseline >= 4.0 else 0.0


def _read(item, asset, bbox):
    da = rxr.open_rasterio(item.assets[asset].href)
    da = da.rio.clip_box(*bbox, crs="EPSG:4326")
    return da.squeeze("band", drop=True)


def load_scene(item, bbox, bands=BANDS, match=None):
    """Return (reflectance DataArray [band, y, x], SCL DataArray [y, x]).

    All bands are resampled to the 10 m B03 grid. Pass `match` (a DataArray)
    to force a common grid across scenes, which permanent-water masks and
    time series require.
    """
    ref = _read(item, "B03", bbox)
    if match is not None:
        ref = ref.rio.reproject_match(match)
    offset = _boa_offset(item)
    layers = []
    for b in bands:
        da = _read(item, b, bbox) if b != "B03" else _read(item, "B03", bbox)
        da = da.rio.reproject_match(ref, resampling=Resampling.bilinear)
        refl = ((da.astype("float32") - offset) / 10000.0).clip(0, 1.5)
        layers.append(refl.expand_dims(band=[b]))
    stack = xr.concat(layers, dim="band")
    scl = _read(item, "SCL", bbox).rio.reproject_match(ref, resampling=Resampling.nearest)
    return stack, scl.astype("uint8")
