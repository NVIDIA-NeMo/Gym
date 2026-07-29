import shlex
from typing import Any


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted ++key.path=value Hydra override tokens."""
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list):
            items = ",".join(str(v) for v in value)
            args.append(shlex.quote(f"+{full_key}=[{items}]"))
        else:
            args.append(shlex.quote(f"+{full_key}={value}"))
    return args
