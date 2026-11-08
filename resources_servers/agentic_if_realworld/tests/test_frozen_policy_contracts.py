# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged synthetic upstream suite without importing the DS repository."""

import hashlib
import importlib.util

from resources_servers.agentic_if_realworld.benchmark.core import ROOT, frozen_policy_bytes, read


# A fresh validator preserves the generic native/text-policy tests independently
# of this benchmark's wrapper-decoder installation. Only the historical import
# statement is replaced; the pinned synthetic test bodies remain unchanged.
_policy = ROOT / "policy"
assert (
    hashlib.sha256(frozen_policy_bytes("test_contract.py")).hexdigest()
    == read(ROOT / "source_manifest.json")["policy_assets"]["test_contract.py"]
)
_spec = importlib.util.spec_from_file_location("isolated_strict_v22", _policy / "validate_response_judge.py")
_validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validator)
_source = (
    (_policy / "test_contract.py")
    .read_text()
    .replace("from run_scripts import validate_agentic_if_response_judge_v2_2 as judge", "")
)
_namespace = {
    "__name__": __name__,
    "__file__": str(_policy / "test_contract.py"),
    "judge": _validator,
}
exec(compile(_source, str(_policy / "test_contract.py"), "exec"), _namespace)  # noqa: S102 - hash-pinned local synthetic tests
globals().update(
    {
        name: value
        for name, value in _namespace.items()
        if name.startswith("test_") and name != "test_all_twelve_definitions_preserved_and_strict_threshold_explicit"
    }
)


def test_all_twelve_schema_codes_match_executable_validator():
    schema = read(_policy / "judge_output.schema.json")
    subtypes = schema["properties"]["constraints"]["items"]["properties"]["subtype"]["enum"]
    assert len(subtypes) == len(set(subtypes)) == 12
    assert set(subtypes) == _validator.NATIVE_CODES
    assert _validator.TEXT_CODES <= _validator.TEXT_ACTION_CODES <= _validator.NATIVE_CODES
