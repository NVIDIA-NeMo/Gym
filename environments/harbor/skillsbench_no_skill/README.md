# SkillsBench (no skills) on the Harbor task shape

The 87 official [SkillsBench](https://github.com/benchflow-ai/skillsbench) tasks as plain Harbor task folders for
the generic `harbor` environment server, without their skills. No task declares `skills_dir` or carries
`environment/skills/`, and the one image that builds skills in runs with an overlay that clears the agent skill
locations. The with-skill condition is the sibling entry, [`skillsbench`](../skillsbench/README.md).

```sh
python -m benchmarks.skillsbench.prepare_harbor_env
gym dataset validate skillsbench_no_skill --sandbox opensandbox --exclude-tasks fix-visual-stability
gym eval run skillsbench_no_skill --agent miniswe_sandboxed_agent --sandbox opensandbox --skip-unloadable \
  --exclude-tasks fix-visual-stability --model-type openai_model --model <model> --model-url <url>
```

`dataset.toml` declares no source: the tasks come from the pinned upstream checkout, not from a Harbor hub, so the
prepare script writes them into this folder (the task folders and `manifest.toml` are gitignored beside this
README), and both commands resolve the entry name to it. Any session harness runs this condition.
[`benchmarks/skillsbench/README.md`](../../../benchmarks/skillsbench/README.md) covers the conditions, the images,
the known limits and the OpenHands path.
