# swe

Any agent harness for swe tasks in a sandbox.
Run by `sandbox_agent`, scored by `anyswe` resources server. 

Prepare data:

```bash
python environments/swe/prepare.py --input-jsonl <swebench_verified.jsonl> --limit 20
```

Start env:
```bash
ng_run "+config_paths=[environments/swe/config.yaml]"
```

Generate rollouts:
```bash
ng_collect_rollouts ... 
```

TODO more content