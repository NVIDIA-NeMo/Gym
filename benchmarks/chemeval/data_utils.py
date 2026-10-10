# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and atomically publish chemeval benchmark rows."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def materialize(
    source: Path, output: Path, *, task_schema: type[BaseModel], identity_key: str, benchmark: str
) -> Path:
    """Validate all rows, then atomically publish a byte-preserving copy.

    This is also used for an explicitly supplied converted input file.
    """
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"Explicit converted {benchmark} input does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as reader, NamedTemporaryFile(dir=output.parent, delete=False) as writer:
            temporary = Path(writer.name)
            seen = set()
            count = 0
            for line_number, line in enumerate(reader, 1):
                try:
                    row = json.loads(line)
                    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
                    if not params.input:
                        raise ValueError("empty model input")
                    task_schema.model_validate(row["verifier_metadata"])
                    identity = row["verifier_metadata"].get(identity_key)
                    if not isinstance(identity, str) or not identity:
                        raise ValueError(f"missing task identity: {identity_key}")
                    if identity in seen:
                        raise ValueError(f"duplicate task identity: {identity}")
                    seen.add(identity)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                writer.write(line)
                count += 1
            if not count:
                raise ValueError(f"Empty benchmark dataset: {source}")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def write_rows(
    rows: Iterable[dict[str, object]], output: Path, *, task_schema: type[BaseModel], identity_key: str, benchmark: str
) -> Path:
    """Publish freshly converted rows using the same schema and identity checks as local imports."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        source = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except BaseException:
            source.unlink(missing_ok=True)
            raise
    try:
        return materialize(source, output, task_schema=task_schema, identity_key=identity_key, benchmark=benchmark)
    finally:
        source.unlink(missing_ok=True)
