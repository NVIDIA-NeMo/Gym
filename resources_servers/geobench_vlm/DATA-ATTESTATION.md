# GEOBench-VLM source and data attestation

This file records provenance and the data-use boundary for this server.

- GEOBench-VLM is approved for evaluation and testing only. Do not train on its questions, options, captions, or images.
- Training alongside this benchmark uses only the eight approved sources listed below, with their attribution.

## Code

- Source: `https://github.com/The-AI-Alliance/GEO-Bench-VLM`
- Pinned commit: `bd07b001cb7e19c83b34988b20e4d6905df077a1`
- Upstream license: Apache-2.0
- Paper: Danish et al., "GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks," ICCV 2025, arXiv:2411.19325

## Dataset

- Hugging Face: `aialliance/GEOBench-VLM`, revision `b3485fe4971213ee13f91960638ddc5c4a098e70`; `Single.parquet` SHA-256 `81bb88e17456a7c32be157a49e07be38f51de24e80c14c0ccf8f3d78f0d15a98`
- Single-image split in this release: 3,211 questions in `Single.parquet`, about 1.4 GB including images
- Additional archives on the dataset repo: `Single.zip`, `Temporal.zip`, `Captioning.zip`, `Ref-Det.zip`, `Ref-Seg.zip`
- The paper reports more than 10,000 manually verified instructions. The released single-image records expand to that count through multiple prompts per question.
- MCQ options and captions were generated with GPT-4o and then manually reviewed. This text is evaluation-only, like the rest of the benchmark.
- Imagery comes from existing remote-sensing collections. Those collections have their own terms. This Gym does not redistribute them.

## What this Gym includes

`data/example.jsonl` is five synthetic questions with a 1-pixel placeholder image so the server, prompt, and letter scorer can be tested without the upstream archive. Rows built from the real dataset (`load_qa_tree`, `materialize_perception`) stay local; `.gitignore` excludes every other `data/*.jsonl`.

## Approved training sources

These are the only datasets approved for training in this line of work. `training_sources.py` holds the same records. This server does not fetch them: `training_sources.initiate` refuses, and a training pipeline that uses a source must carry its attribution and notices through derived records.

| Dataset | Training use | Prepared data | Upstream source(s) | License | Attribution |
| --- | --- | --- | --- | --- | --- |
| BigEarthNet v2 | Multilabel land-cover classification, scene QA, and grounded captions | 6,000 Sentinel-2 RGB patches (B02/B03/B04) with matching reference maps | Modified Copernicus Sentinel-2; CLC2018 / Copernicus Land Monitoring Service reference labels | CDLA-Permissive-1.0 | Credit BigEarthNet v2/reBEN authors; preserve supplied metadata; state "Contains modified Copernicus Sentinel data [year]"; cite the European Union Copernicus Land Monitoring Service and CLC2018 source/DOI. |
| EuroSAT | Ten-class scene classification and class-grounded captions | 27,000 paired RGB and 13-band multispectral patches | Copernicus Sentinel-2 imagery | MIT | Cite Helber et al./EuroSAT; retain the MIT copyright and license notice when redistributing the package or substantial portions; include the applicable Copernicus Sentinel source or modified-data notice. |
| USDA Cropland Data Layer 2020–2023 | Crop and land-cover labels paired with date-compatible imagery | Four national 30 m annual rasters with metadata and overviews | USDA National Agricultural Statistics Service | Public domain; free to redistribute | Credit "USDA National Agricultural Statistics Service, Cropland Data Layer"; preserve the crop year/product metadata and as-is disclaimer; do not imply USDA endorsement. |
| NAIP | High-resolution captions and building/road geometry; later object tasks where labels permit | 89 mosaics across 43 AOIs | USDA FPAC Business Center, Geospatial Enterprise Operations; Planetary Computer and USDA ImageServer acquisitions | Public domain; provider credit requested | Credit USDA FPAC Business Center, Geospatial Enterprise Operations; preserve acquisition year, producer, source/item identifiers where available, bands, and modifications. |
| Microsoft Global ML Building Footprints | Model-derived building polygons joined to eligible NAIP imagery | 550,698 polygon features across 43 AOIs | Microsoft Global ML Building Footprints; underlying imagery rights remain separate | CDLA-Permissive-2.0 for the identified release | Credit Microsoft Global ML Building Footprints; retain license and release evidence, source URLs/index hash, and modification notices; identify the polygons as model-derived. |
| OpenStreetMap | Road, building, water, aeroway, and contextual geometry joined to eligible imagery | Extracts for 43 AOIs | OpenStreetMap contributors | ODbL 1.0 | Use "© OpenStreetMap contributors" and link the ODbL terms; preserve extraction date and source element IDs; retain provenance through derived records. |
| SpaceNet 2 | Building localization and geometry QA | 10,592 complete image/label pairs | SpaceNet Partners and original imagery/annotation contributors | CC BY-SA 4.0 | Credit SpaceNet Partners and the original imagery/annotation contributors; identify modifications and preserve applicable attribution and ShareAlike notices. |
| SpaceNet 3 | Road-centerline localization and geometry QA | 2,549 complete image/label pairs; 231 additional images without labels are audit-only | SpaceNet Partners and original imagery/annotation contributors | CC BY-SA 4.0 | Credit SpaceNet Partners and the original imagery/annotation contributors; identify modifications and preserve applicable attribution and ShareAlike notices. Road centerlines must not be represented as road-surface masks. |
