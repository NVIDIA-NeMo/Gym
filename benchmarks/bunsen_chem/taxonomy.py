# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bunsen Chemistry Taxonomy labels for BunsenBench Chemistry MCQ."""

from __future__ import annotations

from typing import Any


TAXONOMY_VERSION = "bct_gpt55_low_v1"

BCT_SUBFIELDS: dict[str, tuple[str, ...]] = {
    "analytical": (
        "spectroscopy",
        "nmr",
        "mass_spectrometry",
        "chromatography",
        "electroanalytical",
        "titrimetry",
        "gravimetry",
        "qualitative_analysis",
    ),
    "biochemistry": (
        "metabolism",
        "proteins",
        "nucleic_acids",
        "lipids",
        "carbohydrates",
        "molecular_biology",
    ),
    "general": (
        "atomic_structure",
        "bonding",
        "stoichiometry",
        "states_of_matter",
        "solutions",
        "acids_bases",
        "redox",
        "nomenclature",
    ),
    "inorganic": (
        "coordination",
        "main_group",
        "transition_metals",
        "organometallic",
        "solid_state",
        "bioinorganic",
        "nuclear",
    ),
    "materials": (
        "polymers",
        "ceramics",
        "nanomaterials",
        "composites",
        "semiconductors",
        "biomaterials",
        "surfaces",
    ),
    "organic": (
        "structure",
        "stereochemistry",
        "reactions",
        "synthesis",
        "heterocyclic",
        "natural_products",
        "medicinal",
        "polymer",
    ),
    "physical": (
        "thermodynamics",
        "kinetics",
        "quantum",
        "statistical_mechanics",
        "electrochemistry",
        "photochemistry",
        "surface_chemistry",
        "computational",
    ),
    "applied": (
        "industrial_processes",
        "chemical_engineering",
        "green_chemistry",
        "food_chemistry",
        "energy",
        "quality_control",
    ),
    "safety": (
        "toxicology",
        "environmental",
        "hazard_assessment",
        "regulatory",
        "laboratory_safety",
    ),
    "preference": (
        "absorption",
        "distribution",
        "metabolic_stability",
        "excretion",
        "toxicity_prediction",
        "drug_likeness",
    ),
    "not_chemistry": ("not_chemistry",),
}


def normalize_taxonomy_label(label: dict[str, Any]) -> dict[str, str]:
    field = _normalize(label.get("bct_field"))
    subfield = _normalize(label.get("bct_subfield"))
    if field not in BCT_SUBFIELDS:
        raise ValueError(f"Unknown bct_field: {field!r}")
    if subfield not in BCT_SUBFIELDS[field]:
        raise ValueError(f"Unknown bct_subfield for {field!r}: {subfield!r}")
    return {"bct_field": field, "bct_subfield": subfield}


def validate_taxonomy(row: dict[str, Any]) -> None:
    normalize_taxonomy_label(row)


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
