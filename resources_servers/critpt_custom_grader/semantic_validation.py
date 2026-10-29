# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-only evaluation of an operator-owned predicate over inert decoded values.

The runner calls this under its per-case timer and process limits, only in the protected comparator domain.
The controller selects the source from its operator registry. No task or candidate supplies executable source
to this module. Each invocation uses a fresh namespace; validator exceptions never become candidate failures.
"""

from .codec import RepresentationError, decode_value
from .comparator import _result
from .task_data import MAX_DEPTH, MAX_SEMANTIC_CONFIG_BYTES, MAX_VALIDATOR_SOURCE_BYTES, bounded_json


def _value(raw: object) -> object:
    if type(raw) is not dict or set(raw) != {"kind", "value"} or raw["kind"] != "value":
        raise RepresentationError("invalid_outcome")
    return decode_value(raw["value"])


def compare_semantic_request(request: dict, source: str, *, entrypoint: str) -> dict:
    """Apply ``validate(*, observed, expected, args, kwargs, config) -> bool`` without coercion.

    Inputs retain the codec's inert carriers; runner input conversions and equality-comparator tolerances are
    not applied. The operator predicate owns semantic checks and tolerances through its bounded config.
    """
    role = request.get("observed_role")
    if role not in ("reference", "candidate"):
        return _result("uncertain", "request", "invalid_request")
    try:
        if not source.strip() or "\x00" in source or len(source.encode("utf-8")) > MAX_VALIDATOR_SOURCE_BYTES:
            return _result("uncertain", "comparator", "validator_source_invalid")
        try:
            expected = _value(request["expected"])
            inputs = request["inputs"]
            args = [decode_value(value) for value in inputs["args"]]
            kwargs = {name: decode_value(value) for name, value in inputs["kwargs"].items()}
            config = request["validator_config"]
            bounded_json(config, max_bytes=MAX_SEMANTIC_CONFIG_BYTES, max_depth=MAX_DEPTH)
        except RepresentationError as exc:
            return _result("invalid_expected", "expected", exc.code, exc.path)
        try:
            observed = _value(request["observed"])
        except RepresentationError as exc:
            if exc.code == "invalid_outcome":
                return _result("uncertain", role, "invalid_observation_protocol")
            return _result("invalid_" + role, role, exc.code, exc.path)
        namespace = {"__name__": "critpt_operator_validator"}
        exec(compile(source, "<operator-validator>", "exec"), namespace)
        validator = namespace.get(entrypoint)
        if not callable(validator):
            return _result("uncertain", "comparator", "validator_entrypoint_invalid")
        accepted = validator(observed=observed, expected=expected, args=args, kwargs=kwargs, config=config)
        if type(accepted) is not bool:
            return _result("uncertain", "comparator", "validator_return_invalid")
        if accepted:
            return _result("equal", role, "validator_accepted")
        return _result("invalid_reference" if role == "reference" else "mismatch", role, "validator_rejected")
    except TimeoutError:
        return _result("uncertain", "comparator", "comparator_timeout")
    except MemoryError:
        return _result("uncertain", "comparator", "comparator_memory")
    except Exception:
        # Exception text can contain observations or operator source and never crosses this boundary.
        return _result("uncertain", "comparator", "validator_failure")
