# VAB-CSS

The CSS environment of [VisualAgentBench](https://github.com/THUDM/VisualAgentBench) (arXiv:2408.06327): the
model fixes one corrupted CSS rule of a web page from screenshots, in at most 10 turns. Reward is 1.0 when
`1 - SSIM` between the final and the target screenshot is below 0.1. The environment is the
[`vab_css`](../../resources_servers/vab_css/README.md) resources server with `gymnasium_agent`.

## Prepare data

Upstream does not publish the 165-task test split (the repository gitignores `data/css_dataset`), so prepare it
from a local copy and point the resources server at the same folder:

```bash
gym eval prepare --benchmark vab_css \
    +prepare_script_args.data_dir=/path/to/css_dataset
```

The script applies upstream's filter (corruption distance at least 0.2, no invalid examples) and stops unless
165 tasks remain. To prepare another task set, pass `+prepare_script_args.expected_tasks=null`; the two
synthetic tasks in `resources_servers/vab_css/data/example_tasks` also need `+prepare_script_args.min_distance=0`.

## Run

```bash
gym eval run --benchmark vab_css \
    +vab_css_benchmark_resources_server.resources_servers.vab_css.data_dir=/path/to/css_dataset
```

Give `data_dir` as an absolute path: the resources server runs in its own folder. The metric is `mean/reward`
(task success rate). The paper reports 34.5 for gpt-4o-2024-05-13 with prompting.
