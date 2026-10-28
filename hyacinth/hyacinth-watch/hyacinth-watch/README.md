# Hyacinth Watch

Detect and monitor **water hyacinth** on Lake Victoria (Winam Gulf, Kenya) from Sentinel-2, and turn the result into an area/cover time series that fisher communities, researchers and agencies can act on.

Hyacinth mats block landing beaches, fishing grounds and ferry routes. A repeatable, open, cloud-based monitor is cheap to run and easy to scale to other lakes.

## Pipeline

```
STAC search ─► scene load (10 m grid, reflectance) ─► indices (NDVI, FAI, MNDWI)
   ─► water mask ─► weak labels ─► REFINE (QGIS / SAM) ─► chips
   ─► spatial split ─► U-Net ─► scene inference ─► area & cover time series
```

| Step | Notebook | Module |
|---|---|---|
| Data access, first look | `01_data_access` | `stac_data`, `indices` |
| Water mask, weak labels, chips | `02_weak_labels_and_chips` | `weak_labels`, `chips` |
| Train baseline | `03_train_unet` | `model` |
| Inference and time series | `04_inference_and_timeseries` | `inference`, `timeseries` |

## Run it

Colab or Kaggle with a GPU is the easiest path.

```bash
pip install -r requirements.txt
pip install -e .
pytest -q          # sanity-check the core logic
jupyter lab        # open notebooks/01_data_access.ipynb
```

Run the notebooks in order. Settings live in `config/config.yaml`.

## Read this before trusting any number

1. **Verify the bounding boxes.** The pilot and full-gulf boxes in the config are approximate. Check them in notebook 01.
2. **Weak labels are noisy.** Floating cyanobacterial blooms also light up FAI, so threshold labels confuse blooms with hyacinth. Refine labels by hand or with SAM, and report metrics only on a hand-corrected test split. Ground truth from KMFRI (boat or drone surveys) is the best upgrade you can make.
3. **Thresholds are starting points.** Calibrate NDVI/FAI from the histograms in notebook 02.
4. **Water mask.** The image-derived mask can drop chronically infested areas (mats have negative MNDWI). Use a fixed lake polygon (`aoi.water_polygon`) or a surface-water occurrence layer when you can.
5. **Reflectance offset.** Sentinel-2 scenes processed with baseline 04.00 or later (from 2022-01-25) carry a +1000 DN offset; `stac_data.py` handles it, so keep it if you swap data sources.
6. **Spatial split.** Chips are split by geographic block across all dates. Random splits leak and inflate scores.
7. **Not yet run end to end.** The pure-logic modules are unit-tested (`tests/`). The data-access and training code depends on network, GPU and current library versions, so expect to fix small API or version issues on first run.

## Where geoai and GeoAgent fit

The U-Net here is a plain baseline built on `segmentation_models_pytorch`. Next steps:

- Compare against geoai's training utilities and remote-sensing foundation-model backbones (check the geoai docs for current function names).
- Use segment-geospatial (SAM) to speed up label refinement.
- Optionally expose the results through GeoAgent as an "ask the map" layer. Keep it as an interface only; detections and cover numbers should come from the tested pipeline.

## Layout

```
config/       config.yaml
data/         raw, chips, labels/refined, outputs   (git-ignored contents)
docs/         PROJECT_SCOPE.md
models/       trained weights and normalisation stats
notebooks/    01-04
src/hyacinthwatch/   indices, stac_data, weak_labels, chips, model, inference, timeseries
tests/        unit tests for the non-network logic
```
