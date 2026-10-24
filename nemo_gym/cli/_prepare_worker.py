# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import and execute a benchmark preparer only inside its locked environment."""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path


def load_preparer(reference: str):
    """Load a module reference or an explicitly configured absolute script."""
    if not reference.endswith(".py"):
        return importlib.import_module(reference)
    import hashlib

    name = "_gym_preparer_" + hashlib.sha256(reference.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, reference)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preparation script {reference}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--action", choices=("inspect", "prepare"), required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.root)
    module = load_preparer(args.module)
    options = json.loads(args.args)
    if args.action == "inspect":
        current = False
        if options["check_cache"] and args.expected.exists():
            predicate = getattr(module, "is_prepared_data_current", None)
            current = not callable(predicate) or bool(predicate(args.expected))
        result = {"has_prepare": callable(getattr(module, "prepare", None)), "cache_current": current}
    else:
        output = Path(module.prepare(**options)).absolute()
        if output != args.expected:
            raise RuntimeError(f"Prepared dataset path {output} does not match configured path {args.expected}")
        print(f"Dataset prepared at: {output}")
        result = {"output": str(output)}
    args.result.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
