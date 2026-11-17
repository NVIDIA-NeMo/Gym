# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the example dataset and make its PDFs read-only."""

import argparse
import shutil
from pathlib import Path


def prepare_example(output_root: str | Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "resources_servers/pdf_analyst_qa/data/example_bundle"
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
    shutil.copytree(source, root)
    for pdf in (root / "document_store").glob("*.pdf"):
        pdf.chmod(0o444)
    return root / "gym_input.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    prepare_example(**vars(parser.parse_args()))
