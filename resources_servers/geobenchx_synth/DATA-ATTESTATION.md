# GeoBenchX synthetic tasks: source and data attestation

## Sources

All sources are on the approved training-dataset list. They are downloaded by
`prepare/download.py` from their public distribution points and are **not
committed or redistributed** with this repo.

| Source | Use | License | Download |
|---|---|---|---|
| OpenStreetMap | Roads, railways, waterways, points of interest, places, water bodies, county boundaries | ODbL 1.0 | Geofabrik state extracts; Overpass API for county boundaries |
| Microsoft Global ML Building Footprints | Building polygons | CDLA-Permissive-2.0 | `minedbuildings` dataset-links.csv quadkey tiles |
| USDA NASS Cropland Data Layer 2020-2023 | 30 m crop classification rasters | Public domain (US Government) | CropScape `GetCDLFile` per county and year |

## Tools

The tools are the upstream GeoBenchX tools (MIT, copyright 2025 Varvara
Krechetova), pinned to the commit in `../geobenchx/upstream/COMMIT`. The worker
replaces the upstream data catalogs with `catalog.json` and deduplicates spatial
selections (see `synth_worker.py`). No upstream GeoBenchX task, reference
solution, or data snapshot is used.

## Generated content

- Task prompts come from the templates in `templates.py`. No model wrote them.
- Ground truth comes from replaying each template's reference plans through the
  tools on the downloaded data.
- Generated files are built locally and are **not committed**:
  - `GeoData/` and `catalog.json`, from `prepare/build_catalog.py`. This includes
    the per-state crop acreage tables, which are pixel counts from the Cropland
    Data Layer rasters converted to acres.
  - `tasks.jsonl`, `train.jsonl` and `heldout.jsonl`, from `prepare/generate.py`.
  - These files contain dataset names and counts and lengths derived from the
    sources. Treat them as derived data under the source licenses: ODbL share-alike
    and attribution apply to the OpenStreetMap-derived values.

## Splits

Whole states are held out: Georgia and Colorado form the held-out split, and the
other eight states are used for training. Each task's catalog lists only states
from its own split.

## Not used

The GeoBenchX benchmark tasks and their data snapshot are not used for training
here, and neither is GEOBench-VLM.
