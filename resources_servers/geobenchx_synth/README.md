# GeoBenchX synthetic tasks

Multistep GIS tool-use tasks in the style of GeoBenchX, generated from templates
over 40 US counties in 10 states and scored on the final answer rather than by
matching tool calls. The model gets the upstream GeoBenchX tools and a catalog
of OpenStreetMap layers, Microsoft building footprints, USDA Cropland Data
Layer rasters, and a per-state table of crop acreage by county and year derived
from those rasters. See `DATA-ATTESTATION.md` for sources and licenses.

The acreage table's keys are deliberately not ready to join: `County` drops the
word "County", so a join by name matches nothing, while the integer `GEOID`
matches the county layer's `geoid` column.

## Task kinds

- **solve**: the model computes a count or a length with the tools, then ends
  with `Answer: <number>`. Example templates:
  - points of interest of a given type in a county
  - road, railway or waterway length
  - buildings within a distance of a road class
  - buildings on a given crop class in one year
  - the change in that number between 2020 and 2023

  Harder templates chain more steps or leave more to the model:
  - comparisons: the difference between two counties, the county with the most,
    or the year with the most
  - ratios: buildings per km of road, or the percentage of buildings on cropland
  - combined spatial filters: points of interest near both a river and a road
  - inferred: the prompt names neither the dataset nor the class, for example
    "How many buildings stood on land planted with corn in 2022?"

  Table templates use `load_data`, `filter_categorical`, `filter_numerical`
  and `merge_dataframes` on the acreage table: a single value, a sum or mean
  over counties, a change between years, a ratio of two crops, the county
  with the largest change, a comparison with a county in the other offered
  state, and a join to the county layer after one or two spatial selections.
- **reject**: the task needs data that the catalog does not have, such as:
  - a county outside the study set
  - a whole state, when only the study counties are covered
  - a Cropland Data Layer year outside 2020-2023
  - building heights or elevation

  Subtle rejects ask for something the catalog nearly has: a variable with no
  source (traffic volume, building age, crop yield, irrigation), a change over
  time that a single OpenStreetMap snapshot cannot give, a sum that includes a
  county outside the table, or a comparison with a state that is in the data
  but not offered for this task. Each comes with a solvable twin that differs
  only in that detail, so always attempting and always rejecting both lose.

  The model should call `reject_task`.
- **control**: a unit conversion that needs no tools.

## Reward

The reward is binary, computed in `scoring.py`:

- **solve**: 1 if the answer matches one of the accepted ground-truth values
  within the task's tolerance and the model did not reject the task.
- **reject**: 1 if the last tool call is `reject_task`.
- **control**: 1 if the answer is correct and no tool was called.

Ground truth comes from replaying every reference plan of a task through the
real tools. Some templates have several valid plans; for example, a county can
be selected by its attribute or by its geometry. Each plan's result is an
accepted answer.

## Build

Run these steps with the GeoBenchX geospatial environment (`../geobenchx/tools_env`):

```bash
cd resources_servers/geobenchx_synth/prepare
python download.py --out $RAW
python build_catalog.py --raw $RAW --out $GEODATA
python generate.py --geodata $GEODATA --upstream-src $GEOBENCHX_SRC --out $TASKS
```

`generate.py` keeps a task only if its first plan succeeds with a nonzero result
and every plan runs within the time limit. It prints what it dropped and why.

## Run

Set the paths listed at the top of `configs/geobenchx_synth.yaml` in `env.yaml`, then:

```bash
gym env start "+config_paths=[resources_servers/geobenchx_synth/configs/geobenchx_synth.yaml,<model config>]"
```

Train on `train.jsonl` only. `heldout.jsonl` covers Georgia and Colorado, which
never appear in training catalogs.
