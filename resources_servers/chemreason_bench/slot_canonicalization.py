# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-completion slot canonicalization, ported from upstream's prediction side.

Derived from ``predict/predict.py`` in https://github.com/Khadaz/ChemReason-Bench
at commit ``c0b9ac2933708fcca47b1795492952cbf280e194`` (Apache-2.0), which runs
``canonicalize_slots(slots, legend)`` at line 953 before a step-completion
prediction reaches the scorer.

Without this the scorer sees raw model keys. An answer upstream credits as a
matched reagent is instead counted as an unrecognised extra key, or -- worse --
as an illegal unit, which sets the fatal flag and zeroes the whole task through
the corpus-level format-error penalty. ``step_completion_score`` is one sixth of
Primary-Overall, so the effect is not marginal.

Gold-as-prediction cannot detect the omission, because gold slots are already
canonical. Only a model reply exercises this path.

NOTE ON THE UNIT MAPS. Upstream keeps two different UCUM tables: the one in
``predict.py`` used here additionally maps concentration units (``M``, ``N``),
while ``eval.py``'s -- reproduced in ``metrics.py`` -- does not. They are
deliberately not merged: canonicalization and scoring are separate stages and
upstream applies a different table at each.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# predict.py's DEFAULT_UCUM. Differs from eval.py's by the concentration entries.
_UCUM = {
    # volume
    "μl": "uL",
    "µl": "uL",
    "ul": "uL",
    "μL": "uL",
    "µL": "uL",
    "uL": "uL",
    "ml": "mL",
    "mL": "mL",
    "l": "L",
    "L": "L",
    "ul.": "uL",
    "liter": "L",
    "litre": "L",
    "ltrs": "L",
    # mass
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "µg": "ug",
    "μg": "ug",
    "ug": "ug",
    "ng": "ng",
    # substance amount
    "mol": "mol",
    "mmol": "mmol",
    "µmol": "umol",
    "μmol": "umol",
    "umol": "umol",
    "mole": "mol",
    "moles": "mol",
    # time
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    # temperature
    "°c": "C",
    "c": "C",
    "°f": "F",
    "f": "F",
    "k": "K",
    "C": "C",
    "F": "F",
    "K": "K",
    # concentration -- present here, absent from eval.py's table
    "m": "M",
    "M": "M",
    "n": "N",
    "N": "N",
}

TIME_UNITS = {"s", "min", "h"}
TEMP_UNITS = {"C", "F", "K"}
_MASS_VOLUME_MOLE_UNITS = {"mg", "g", "kg", "ug", "ng", "uL", "mL", "L", "mol", "mmol", "umol"}

ALIASES_TO_REAGENT = {
    "reactant",
    "substrate",
    "nucleophile",
    "base",
    "acid",
    "solvent",
    "reagent",
    "chemical",
    "chemical_id",
    "with_chemical",
    "wash_with",
    "drying_agent",
    "solid_dry_with",
    "dry_agent",
    "dry_with",
}

SLOT_WHITELIST = {
    "reagent",
    "amount_value",
    "amount_unit",
    "duration_value",
    "duration_unit",
    "temperature_value",
    "temperature_unit",
    "duration_token",
    "temperature_token",
}

PLACEHOLDER_RE = re.compile(r"\$-?\d+\$")
_AMOUNT_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z°µμ]+)")
_TEMP_TOKEN_RE = re.compile(r"(#\d+#)")
_DUR_TOKEN_RE = re.compile(r"(@\d+@)")


def norm_unit(unit: Any) -> Any:
    if not isinstance(unit, str):
        return unit
    return _UCUM.get(unit.strip().lower(), unit.strip())


def parse_amount_blob(text: Any) -> Optional[Tuple[float, str]]:
    """Split a blob like ``"10 mL"`` into ``(10.0, "mL")``."""
    if not isinstance(text, str):
        return None
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    return float(match.group(1)), norm_unit(match.group(2))


def _normalize_name(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\+\-\[\]\.,/ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_legend_maps(legend: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    for placeholder, name in (legend or {}).items():
        if not isinstance(placeholder, str) or not PLACEHOLDER_RE.fullmatch(placeholder):
            continue
        id2name[placeholder] = name
        normalized = _normalize_name(name)
        if normalized:
            name2id[normalized] = placeholder
    return id2name, name2id


def to_placeholder(value: Any, name2id: Dict[str, str]) -> Any:
    """Map a reagent NAME back to its ``$n$`` placeholder using the legend."""
    if isinstance(value, str):
        match = PLACEHOLDER_RE.search(value)
        if match:
            return match.group(0)
        normalized = _normalize_name(value)
        if normalized in name2id:
            return name2id[normalized]
    return value


def _migrate_amount_to_duration(out: Dict[str, Any]) -> None:
    if out.get("amount_unit") in TIME_UNITS:
        out["duration_value"] = out.get("amount_value")
        out["duration_unit"] = out.get("amount_unit")
        out.pop("amount_value", None)
        out.pop("amount_unit", None)


def _sanitize_units(out: Dict[str, Any]) -> None:
    for key in ("amount_unit", "duration_unit", "temperature_unit"):
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, list):
            value = value[0] if value else ""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:  # pragma: no cover - str() of an arbitrary object
                out.pop(key, None)
                continue
        value = norm_unit(value)
        if not value:
            out.pop(key, None)
        else:
            out[key] = value

    unit = out.get("amount_unit")
    if unit is None:
        return
    if not isinstance(unit, str):
        out.pop("amount_value", None)
        out.pop("amount_unit", None)
        return
    if unit not in _MASS_VOLUME_MOLE_UNITS:
        if unit in TIME_UNITS:
            _migrate_amount_to_duration(out)
        elif unit in TEMP_UNITS:
            out["temperature_value"] = out.get("amount_value")
            out["temperature_unit"] = unit
            out.pop("amount_value", None)
            out.pop("amount_unit", None)
        else:
            out.pop("amount_value", None)
            out.pop("amount_unit", None)


def canonicalize_slots(slots: Optional[Dict[str, Any]], legend: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Port of upstream ``canonicalize_slots`` (predict.py:625).

    Alias keys collapse to ``reagent``; a placeholder used as a KEY becomes the
    reagent with its value parsed as an amount; blob-valued ``amount`` /
    ``temperature`` / ``time`` keys split into value+unit pairs; reagent names
    resolve to ``$n$`` via the legend; units are normalised and migrated between
    groups; ``#n#`` / ``@n@`` tokens are lifted out of any value; and anything
    outside the whitelist is dropped.
    """
    _, name2id = build_legend_maps(legend)

    out: Dict[str, Any] = {}
    for key, value in dict(slots or {}).items():
        if isinstance(key, str) and key.lower() in ALIASES_TO_REAGENT:
            key = "reagent"
        out[key] = value

    # A placeholder used as a key, e.g. {"$4$": "10 mL"}.
    for key in list(out.keys()):
        if isinstance(key, str) and PLACEHOLDER_RE.fullmatch(key):
            value = out.pop(key)
            out["reagent"] = key
            parsed = parse_amount_blob(str(value))
            if parsed:
                out["amount_value"], out["amount_unit"] = parsed

    for key in list(out.keys()):
        low = key.lower() if isinstance(key, str) else ""
        if low in {"amount", "volume", "mass", "quantity", "qty", "dose"} and isinstance(out[key], str):
            parsed = parse_amount_blob(out[key])
            if parsed:
                out["amount_value"], out["amount_unit"] = parsed
            out.pop(key, None)
        elif low in {"temperature", "temp"} and isinstance(out[key], str):
            parsed = parse_amount_blob(out[key])
            if parsed:
                out["temperature_value"], out["temperature_unit"] = parsed
            out.pop(key, None)
        elif low in {"time", "duration"} and isinstance(out[key], str):
            parsed = parse_amount_blob(out[key])
            if parsed:
                out["duration_value"], out["duration_unit"] = parsed
            out.pop(key, None)
        elif low == "settemperature":
            parsed = parse_amount_blob(str(out[key]))
            if parsed:
                out["temperature_value"], out["temperature_unit"] = parsed
            out.pop(key, None)

    for key in list(out.keys()):
        if isinstance(key, str) and (key == "reagent" or key.startswith("reagent_")):
            out[key] = to_placeholder(out[key], name2id)

    _sanitize_units(out)
    _migrate_amount_to_duration(out)

    for numeric_key in ("amount_value", "duration_value", "temperature_value"):
        if numeric_key in out:
            try:
                out[numeric_key] = float(out[numeric_key])
            except (TypeError, ValueError, OverflowError):
                # OverflowError, not ValueError, is what a several-hundred-digit
                # JSON integer raises; unhandled it escapes verify() as a 500.
                out.pop(numeric_key, None)

    temperature_token = duration_token = None
    for value in list(out.values()):
        if isinstance(value, str):
            match = _TEMP_TOKEN_RE.search(value)
            if match:
                temperature_token = match.group(1)
            match = _DUR_TOKEN_RE.search(value)
            if match:
                duration_token = match.group(1)
    if temperature_token:
        out["temperature_token"] = temperature_token
        out.pop("temperature_value", None)
        out.pop("temperature_unit", None)
    if duration_token:
        out["duration_token"] = duration_token
        out.pop("duration_value", None)
        out.pop("duration_unit", None)

    cleaned: Dict[str, Any] = {}
    for key, value in out.items():
        if not isinstance(key, str):
            continue
        if key == "reagent" or key.startswith("reagent_") or key in SLOT_WHITELIST:
            cleaned[key] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    return cleaned
