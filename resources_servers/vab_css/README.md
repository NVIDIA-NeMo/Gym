# VAB-CSS

The CSS environment of VisualAgentBench ([arXiv:2408.06327](https://arxiv.org/abs/2408.06327),
[THUDM/VisualAgentBench](https://github.com/THUDM/VisualAgentBench), Apache-2.0) as a Gymnasium-style
resources server. Use it with `gymnasium_agent`.

The model gets a target screenshot, the page's HTML and the current screenshot of a page with one corrupted CSS
rule. It fixes the CSS with one text action per turn, for at most 10 turns:

- `get_selectors_by_html_elements("<BeautifulSoup find_all arguments>")`
- `select_rule('<selector>')`
- `edit_rule('<selector>', '<property>', '<value>')`, which returns a new screenshot
- `revert_last_edit()`

The episode ends when the model writes "I have fixed the css style" or the turns run out. The last screenshot
is then compared with the target: reward is 1.0 when `1 - SSIM < 0.1` (grayscale SSIM), else 0.0. `info` also
has `distance`, `initial_distance` and `improved` (the distance fell by more than 0.05), upstream's secondary
metric. The prompt, the action parsing, the tool messages and the score follow upstream's
`src/server/tasks/css_agent` at commit `9055fc29`. Two differences:

- `get_selectors_by_html_elements` accepts literal arguments only; upstream passes them to `eval()`.
- `insert_rule` is not an action. Upstream's call always fails (it passes the method a string), and the prompt
  does not offer it; calling it here gets upstream's "is not valid due to exception" message too.

Like upstream, `revert_last_edit` takes no new screenshot, so the score uses the last screenshot taken.

## Data

Each task is a folder in upstream's `data/css_dataset` layout: `index.html`, its stylesheets (corrupted),
`index.png` (the target), and `corruption/` with `index_corrupted.png`, `distance.txt` and an optional
`instruction.txt`. Set `data_dir` to the folder of task folders. The 165-task test split is not published
upstream; `data/example_tasks` holds two synthetic tasks for tests and smoke runs.

Build the JSONL from a task folder with upstream's filter (corruption distance at least 0.2, no invalid
examples):

```bash
python resources_servers/vab_css/prepare_data.py --data-dir <css_dataset> --output <file>.jsonl
```

## Run

The server installs Playwright Chromium at startup (`install_browser: true`). Pages render at Playwright's
default 1280x720 viewport, full page. Upstream checks that `cinema-world/index.html` renders at 1280x1421.
The CSS is model-written, so a page may load only files in its own task folder: every network request is
blocked and logged, which also names any remote asset a task would need.

```bash
gym env start --resources-server vab_css --model-type vllm_model
gym eval run --no-serve --agent vab_css_gymnasium_agent \
    --input resources_servers/vab_css/data/example.jsonl --output results/vab_css_rollouts.jsonl
```
