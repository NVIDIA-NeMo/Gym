# Built-in environment suites

A **suite** is a named list of environment names that expands to the `config_paths` needed to run
them together — so a multi-environment sweep becomes `gym run --suite <name>` instead of an 80+
entry hand-maintained `config_paths`.

Each suite is a YAML file named `<suite_name>.yaml`:

```yaml
description: Ultra V3 evaluation suite
environments:
  - gpqa
  - workplace_assistant
```

- `environments` — environment names resolved against the repo's `environments/` directory (see
  `nemo_gym/registry.py`). An unknown name fails fast with a "did you mean?" hint.
- `description` — optional, shown by `gym list suites`.

Files here ship as **read-only built-in** suites. Users add their own under
`~/.config/nemo_gym/suites/`, which shadow a built-in of the same name.

Built-in suites (e.g. `ultra_v3`, `reasoning`, `coding`) are added here as the corresponding
environments are migrated into `environments/`; a suite only references environments that exist.
