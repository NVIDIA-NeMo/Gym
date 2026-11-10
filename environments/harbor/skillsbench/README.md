# SkillsBench (with skills) on the Harbor task shape

The 87 official [SkillsBench](https://github.com/benchflow-ai/skillsbench) tasks as plain Harbor task folders for
the generic `harbor` environment server, with each task's skills. Every task declares
`skills_dir = "/skills"`, and the server uploads the task's `environment/skills/` there at seed. The
no-skill condition is the sibling entry, [`skillsbench_no_skill`](../skillsbench_no_skill/README.md).

```sh
python -m benchmarks.skillsbench.prepare_harbor_env
gym dataset validate skillsbench --sandbox opensandbox --exclude-tasks fix-visual-stability
gym eval run skillsbench --agent miniswe_sandboxed_agent --sandbox opensandbox --skip-unloadable \
  --exclude-tasks fix-visual-stability --model-type openai_model --model <model> --model-url <url>
```

`dataset.toml` declares no source: the tasks come from the pinned upstream checkout, not from a Harbor hub, so the
prepare script writes them into this folder (the task folders and `manifest.toml` are gitignored beside this
README), and both commands resolve the entry name to it. Only a harness that declares `supports_task_tool_context`
runs this condition. [`benchmarks/skillsbench/README.md`](../../../benchmarks/skillsbench/README.md) covers the
conditions, the images, the known limits and the OpenHands path.
