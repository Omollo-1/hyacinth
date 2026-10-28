# Project scope: Hyacinth Watch

## Problem
Water hyacinth repeatedly infests the Winam Gulf and other parts of Lake Victoria, disrupting fishing, transport, water intake and lake ecology. Monitoring is mostly episodic and field-based, so extent and change are poorly documented between surveys.

## Objective
Build an open, reproducible pipeline that maps hyacinth extent from Sentinel-2 at 10 m, tracks area and cover over time, and delivers results in a form local stakeholders can use.

## Users and candidate partners
- KMFRI (existing connection): scientific validation, field or drone ground truth, and a route to users.
- Beach management units and fisher groups: where mats are, and whether landing sites are blocked.
- County and national agencies, and lake-basin bodies: trend evidence for control operations.
- Academic collaborators: methods and publication.

Partners above are candidates, not commitments. Confirm interest before listing anyone as a collaborator.

## Scope
**In:** Kisumu Bay pilot, 2023-2025 Sentinel-2 L2A, binary hyacinth segmentation, monthly cover time series, GeoTIFF and CSV outputs, a simple map viewer.
**Out for now:** species-level discrimination, forecasting, the full gulf (phase 2), and operational alerts (phase 3).

## Method summary
Sentinel-2 via STAC, FAI and NDVI over a water mask for weak labels, refinement by hand or SAM, U-Net baseline, spatial block cross-split, sliding-window inference, per-scene and monthly statistics.

## Validation plan
- Hand-label a held-out set of chips across seasons and locations (target a few hundred chips, including bloom look-alikes).
- Report IoU, F1, precision and recall on that set only, plus per-season breakdown.
- Compare against the threshold-only baseline to show what the model adds.
- Field check: overlay detections with KMFRI observations or drone imagery on at least a few dates.

## Phases and indicative timeline
| Weeks | Milestone |
|---|---|
| 1 | Environment, data access, bbox check (notebook 01) |
| 2-3 | Water mask, weak labels, threshold calibration, first chips |
| 3-5 | Label refinement (SAM-assisted), hand-labelled test set |
| 5-6 | Train and evaluate baseline; error analysis (blooms, glint, shoreline) |
| 6-7 | Full-period inference, time series, maps |
| 8 | Write-up, simple dashboard, partner feedback |
| Phase 2 | Full Winam Gulf, foundation-model backbones, SAR (Sentinel-1) for cloud gaps |
| Phase 3 | Near-real-time updates and alerts to stakeholders |

## Main risks
| Risk | Mitigation |
|---|---|
| Bloom vs hyacinth confusion | Bloom examples in labels; NDVI/texture features; field validation |
| Cloud gaps over the lake | Keep observed-fraction filter; add Sentinel-1 in phase 2 |
| Weak-label bias inherited by the model | Hand-corrected test set; report against it only |
| Sun glint and shoreline mixed pixels | Glint-aware filtering; exclude a thin shoreline buffer; error analysis |
| Overfitting to one bay | Spatial split; test on a second bay before claiming generality |
| Compute limits | Colab/Kaggle GPU; pilot AOI first |

## Success criteria
- Test IoU meaningfully above the threshold baseline on hand-labelled chips.
- A 2023-2025 monthly cover series for the pilot that matches field observations qualitatively.
- One partner (for example KMFRI) reviews the outputs and gives feedback.
- Code, notebooks and outputs reproducible by someone else from the README.

## Outputs
Probability rasters (GeoTIFF), monthly area and cover CSV, figures, a short technical note, and this repository.
