# Description
Thin wrapper around the SynClaw retrosynthesis planning tools.

Build the default SynClaw database (reaction templates, eMolecules plus common reagents, search
index) with `python resources_servers/synclaw/prepare_db.py`; see
[`environments/synclaw/README.md`](../../environments/synclaw/README.md#building-the-synclaw-database).

Data links: [SynClaw](https://github.com/NVIDIA-BioNeMo/SynClaw),
see the *Task sets* section of `README.md`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- synclaw: Apache 2.0

By default (`hide_target_building_block: true` in the resources-server config), all
stock entries identical to each task target are excluded from searches, lookup and
pathway grounding, so the target cannot be submitted as a direct purchase. Set it to
`false` to keep the target in stock. The harness supplies the target; the model cannot
change this setting. Other tasks retain their own stock view, and synthesis of the
target remains valid. Scored rows record `target_hidden_from_stock` for replay and
analysis.
