# Apex Agents rollout agent

This agent runs the same Stirrup 0.1 harness family used by Gym's GDPval agent inside a pinned Archipelago image.
Archipelago owns the world, workplace MCP servers, and `/apps` + `/mcp/` gateway. Stirrup owns the model/tool loop.
The Apex harness repository is not cloned or imported.

Each rollout:

1. restores the cached world ZIP into `/filesystem` and `/.apps_data`;
2. starts Archipelago's gateway and configures the MCP servers packaged in the image;
3. saves the initial local-grader ZIP snapshot;
4. runs Stirrup for at most 200 model turns;
5. saves the final ZIP snapshot; and
6. sends completed submissions, their final answer, and both snapshots to the resources server for grading.

The initial Stirrup toolbelt contains `list_tools`, `inspect_tool`, `add_tool`, `remove_tool`, `todo_write`, and
`finish`. MCP tools are absent until the model adds them. Adding/removing a tool changes the schemas sent on the next
model turn. MCP calls time out after 60 seconds. Long text output uses a 20,000-character head plus 5,000-character
tail excerpt; Stirrup compresses image blocks to about one megapixel when serializing them for the model.

`finish` is the only submission mechanism. A completed submission is rejected while any todo is pending or in
progress. A max-turn exit or `status="incomplete"` does not call the grader; its initial/final snapshots are still
saved under the configured artifact output directory for inspection.

The Archipelago image is reused when present. If it is missing and automatic building is enabled, Gym exports the
configured pinned Archipelago commit and builds the SIF once. The lightweight Stirrup runtime is pinned in
`stirrup-requirements.txt`, built once inside that image, and cached under `deps/`; it does not vendor either source
repository into Gym.
