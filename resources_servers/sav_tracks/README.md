# sav_tracks

Verifier for the SA-V `{"tracks":[...]}` answer schemas (the sav_{pt,box,ref}_rl
jsonls, whose prompts pair with SFT data — do not change the expected answer
shapes). One server scores three task types, selected per record via `task`:

| task | answer entry | GT-visible reward |
|------|--------------|-------------------|
| `pt` | `{"point":i,"frame":k,"xy":[x,y],"visible":bool}` | 1.0 iff point inside the object's segmentation mask |
| `box` | `{"box":i,"frame":k,"bbox":[x1,y1,x2,y2],"visible":bool}` | IoU |
| `ref` | `{"id":i,"frame":k,"xy":[x,y]}` (omission = invisible, ids model-chosen) | 1.0 iff point inside mask, after one-to-one id assignment |

All coordinates are integers on a 0-1000 grid. Ground truth lives in `objects`
(per-object `targets`; visible pt/ref targets carry a COCO compressed-RLE mask,
decoded with a dependency-free pure-Python membership check). GT-invisible
frames pay `absence_score` for a correct invisibility claim, 0 for any
location. Per-frame credit is averaged per object then across objects; extra,
duplicate, or invalid entries scale the reward by `targets/(targets+extras)`;
format failures score 0.

Data is produced by
`tracking/data_utils/convert_sav_rl_to_tracks.py` (outside this repo) from the
portable SA-V RL jsonls, matching each record back to its GT masklets.
Agents: `sav_pt_tracks_agent`, `sav_box_tracks_agent`, `sav_ref_tracks_agent`.
