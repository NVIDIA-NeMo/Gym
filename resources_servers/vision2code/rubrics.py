# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions Copyright 2026 Vision2Code Authors (Apache-2.0).
# Focused port; see NOTICE and tests/fixtures/provenance.json.
from __future__ import annotations

import base64
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageStat


RUBRIC_VERSION = "recreation_rubric_v5_whitespace_diagnostic"

SCORE_MIN = 0.0

SCORE_MAX = 5.0


def _category(category_id: str, label: str, weight: float, description: str) -> Dict[str, Any]:
    return {
        "id": category_id,
        "label": label,
        "weight": weight,
        "description": description,
    }


def _extra_cap(
    cap_type: str,
    cap_to: float,
    reason: str,
    *,
    all_of: Optional[Sequence[Mapping[str, float]]] = None,
    any_of: Optional[Sequence[Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    return {
        "type": cap_type,
        "cap_to": cap_to,
        "reason": reason,
        "all_of": list(all_of or []),
        "any_of": list(any_of or []),
    }


STRICT_CAP_PROFILE = {
    "critical_any_threshold": 1.5,
    "critical_any_cap": 2.5,
    "critical_pair_threshold": 2.5,
    "critical_pair_cap": 3.5,
    "top_score_trigger": 4.8,
    "top_score_min_critical": 4.6,
    "top_score_cap": 4.5,
}

BALANCED_CAP_PROFILE = {
    "critical_any_threshold": 1.5,
    "critical_any_cap": 2.8,
    "critical_pair_threshold": 2.5,
    "critical_pair_cap": 4.0,
    "top_score_trigger": 4.8,
    "top_score_min_critical": 4.4,
    "top_score_cap": 4.5,
}

HARD_CAP_PROFILE = {
    "critical_any_threshold": 1.2,
    "critical_any_cap": 3.0,
    "critical_pair_threshold": 2.2,
    "critical_pair_cap": 4.0,
    "top_score_trigger": 4.8,
    "top_score_min_critical": 4.2,
    "top_score_cap": 4.5,
}

DATASET_RUBRICS: Dict[str, Dict[str, Any]] = {
    "ChartQA": {
        "dataset": "ChartQA",
        "guidance": (
            "This is an easy chart recreation dataset. Be strict about preserving both the information and the chart "
            "structure. Wrong chart orientation, wrong chart type, crowded labels, illegible text, bad title "
            "placement, or legend overlap should all be penalized."
        ),
        "prompt_notes": [
            "Preserve recreation structure, not just semantic content.",
            "If a horizontal chart becomes vertical, preserve some credit for correct data but do not award a perfect score.",
            "Color scheme matters less than readable text, correct numbers, and faithful structure.",
        ],
        "categories": [
            _category(
                "numeric_information_fidelity",
                "Numeric Information Fidelity",
                0.30,
                "Values, trends, ordering, and answer-relevant chart information are correct.",
            ),
            _category(
                "structure_layout_fidelity",
                "Structure and Layout Fidelity",
                0.30,
                "Chart type, orientation, title placement, legend placement, and overall chart structure match the reference.",
            ),
            _category(
                "text_label_legibility_accuracy",
                "Text, Label, and Legibility Accuracy",
                0.25,
                "Labels, numbers, headings, and text remain accurate, legible, and not crowded.",
            ),
            _category(
                "cleanliness_completeness",
                "Cleanliness and Completeness",
                0.15,
                "The chart is unclipped, readable, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "chart_orientation_or_type_mismatch",
                3.5,
                "A wrong chart orientation or chart type cannot receive full credit even when the core numbers are preserved.",
                all_of=[
                    {"id": "numeric_information_fidelity", "min": 3.0},
                    {"id": "structure_layout_fidelity", "max": 2.9},
                ],
            ),
            _extra_cap(
                "material_text_or_numeric_error",
                2.5,
                "Material numeric or text errors cap the score on this dataset.",
                any_of=[
                    {"id": "numeric_information_fidelity", "max": 2.4},
                    {"id": "text_label_legibility_accuracy", "max": 2.4},
                ],
            ),
        ],
    },
    "dvqa": {
        "dataset": "dvqa",
        "guidance": (
            "This is an easy chart recreation dataset. Be strict about exact plotted information, faithful structure, "
            "and readable labels. Preserve the same chart structure rather than only approximate semantics."
        ),
        "prompt_notes": [
            "Wrong orientation, wrong chart structure, or poor label legibility should clearly lower the score.",
            "Legend placement or labels crowding the plot should be penalized.",
            "Colors matter less than correct information, structure, and readable text.",
        ],
        "categories": [
            _category(
                "numeric_information_fidelity",
                "Numeric Information Fidelity",
                0.30,
                "Values, trends, grouping, and answer-relevant chart information are correct.",
            ),
            _category(
                "structure_layout_fidelity",
                "Structure and Layout Fidelity",
                0.30,
                "Chart type, grouping or stacking structure, orientation, title placement, and legend placement match the reference.",
            ),
            _category(
                "text_label_legibility_accuracy",
                "Text, Label, and Legibility Accuracy",
                0.25,
                "Axis labels, category labels, numbers, and titles remain accurate, legible, and not crowded.",
            ),
            _category(
                "cleanliness_completeness",
                "Cleanliness and Completeness",
                0.15,
                "The chart is unclipped, readable, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "chart_orientation_or_type_mismatch",
                3.5,
                "A wrong chart orientation or chart type cannot receive full credit even when the core numbers are preserved.",
                all_of=[
                    {"id": "numeric_information_fidelity", "min": 3.0},
                    {"id": "structure_layout_fidelity", "max": 2.9},
                ],
            ),
            _extra_cap(
                "material_text_or_numeric_error",
                2.5,
                "Material numeric or text errors cap the score on this dataset.",
                any_of=[
                    {"id": "numeric_information_fidelity", "max": 2.4},
                    {"id": "text_label_legibility_accuracy", "max": 2.4},
                ],
            ),
        ],
    },
    "figureqa": {
        "dataset": "figureqa",
        "guidance": (
            "This is an easy chart recreation dataset. Be strict about preserving the same structure, readable labels, "
            "and correct plotted information. Semantic preservation alone is not enough for a top score."
        ),
        "prompt_notes": [
            "If the data are right but the plot structure changes meaningfully, keep partial credit but not full credit.",
            "Crowded or illegible labels should be penalized.",
            "Minor color differences matter less than correct structure and readable information.",
        ],
        "categories": [
            _category(
                "numeric_information_fidelity",
                "Numeric Information Fidelity",
                0.30,
                "Values, trends, series relationships, and answer-relevant information are correct.",
            ),
            _category(
                "structure_layout_fidelity",
                "Structure and Layout Fidelity",
                0.30,
                "Plot type, orientation, legend placement, series structure, and overall chart layout match the reference.",
            ),
            _category(
                "text_label_legibility_accuracy",
                "Text, Label, and Legibility Accuracy",
                0.25,
                "Legends, labels, headings, and textual chart content remain accurate and readable.",
            ),
            _category(
                "cleanliness_completeness",
                "Cleanliness and Completeness",
                0.15,
                "The chart is unclipped, readable, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "chart_orientation_or_type_mismatch",
                3.5,
                "A wrong chart orientation or chart type cannot receive full credit even when the core numbers are preserved.",
                all_of=[
                    {"id": "numeric_information_fidelity", "min": 3.0},
                    {"id": "structure_layout_fidelity", "max": 2.9},
                ],
            ),
            _extra_cap(
                "material_text_or_numeric_error",
                2.5,
                "Material numeric or text errors cap the score on this dataset.",
                any_of=[
                    {"id": "numeric_information_fidelity", "max": 2.4},
                    {"id": "text_label_legibility_accuracy", "max": 2.4},
                ],
            ),
        ],
    },
    "ChemVQA-2K": {
        "dataset": "ChemVQA-2K",
        "guidance": (
            "Focus on chemistry semantics: molecular structure, chemical symbols, bond lines, and important colored "
            "difference markers. Missing a non-core red difference marker should hurt moderately, not collapse the score."
        ),
        "prompt_notes": [
            "Atom symbols such as O, OH, Cl, charges, and bond identities are critical.",
            "Preserve important colored markers or difference lines when they are semantically meaningful.",
            "Small orientation changes matter less than structure and symbol correctness.",
        ],
        "categories": [
            _category(
                "molecular_structure_fidelity",
                "Molecular Structure Fidelity",
                0.40,
                "Connectivity, overall structure, and core arrangement of the molecule are correct.",
            ),
            _category(
                "chemical_symbol_accuracy",
                "Chemical Symbol Accuracy",
                0.30,
                "Atom labels, charges, and bond notation remain accurate and readable.",
            ),
            _category(
                "bond_line_marker_fidelity",
                "Bond Line and Marker Fidelity",
                0.20,
                "Bond lines, highlighting, color-coded difference markers, and related visual indicators are preserved.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.10,
                "The structure is readable, unclipped, and not missing major visible content.",
            ),
        ],
        "cap_profile": BALANCED_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "structure_or_symbol_failure",
                2.5,
                "Wrong molecular structure or wrong chemical symbols cap the score on this dataset.",
                any_of=[
                    {"id": "molecular_structure_fidelity", "max": 2.4},
                    {"id": "chemical_symbol_accuracy", "max": 2.4},
                ],
            ),
        ],
    },
    "DocVQA": {
        "dataset": "DocVQA",
        "guidance": (
            "Judge text and structure first. Be strict, but stay fair on hard infographics: if the source itself is "
            "visually dense or difficult, do not over-penalize ambiguity that is already present in the reference."
        ),
        "prompt_notes": [
            "Visible text should be accurate and legible if the source is legible.",
            "Layout should stay close to the reference, but exact decorative styling matters less than information and structure.",
            "Visible layout drift should lower the layout score even when the text is exact.",
            "Match background, text, and graphic colors to the reference; visible color mismatches should lower the key_visual_elements score.",
        ],
        "categories": [
            _category(
                "text_information_fidelity",
                "Text Information Fidelity",
                0.35,
                "Visible text, numbers, and key document information remain accurate.",
            ),
            _category(
                "document_structure_layout",
                "Document Structure and Layout",
                0.30,
                "Reading order, section placement, tables, blocks, and layout structure stay faithful.",
            ),
            _category(
                "legibility_noncrowding",
                "Legibility and Non-Crowding",
                0.20,
                "Content remains readable, not overly crowded, and reasonably well spaced.",
            ),
            _category(
                "key_visual_elements",
                "Key Visual Elements",
                0.15,
                "Important icons, tables, charts, callouts, or emphasis boxes remain present when they matter.",
            ),
        ],
        "cap_profile": BALANCED_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "material_key_text_error",
                2.5,
                "Materially wrong key text or numbers cap the score on this dataset.",
                any_of=[
                    {"id": "text_information_fidelity", "max": 2.4},
                ],
            ),
        ],
    },
    "EEE-Bench": {
        "dataset": "EEE-Bench",
        "revision": "eee_background_v1",
        "guidance": (
            "Focus on circuit topology, component identity, labels, values, signs, and current or voltage direction. "
            "Be lenient on stylistic drawing differences such as zig-zag versus curved resistor style when the component "
            "identity is still clearly correct. Background color is a separate fidelity requirement; don't ignore it."
        ),
        "prompt_notes": [
            "Wrong direction, sign, label, or connectivity matters more than cosmetic symbol style.",
            "Component identity must be preserved even if the drawn shape style varies slightly.",
            "Treat small labels and values as potentially important.",
            "Compare the background across the canvas, margins, and any distinct panel regions. "
            "Preserve the reference hue, brightness, and tint; do not assume the background should be white.",
            "For background_color_fidelity, award 5 only when no background color mismatch is visible. "
            "A visible white-versus-gray, cream, tinted, or dark background mismatch must score below 5, "
            "even if the circuit and labels are otherwise perfect. Ignore only imperceptible compression noise.",
        ],
        "categories": [
            _category(
                "circuit_topology_connectivity",
                "Circuit Topology and Connectivity",
                0.30,
                "Wiring relationships, placement relationships, and the overall circuit topology are correct.",
            ),
            _category(
                "component_identity_symbols",
                "Component Identity and Symbols",
                0.25,
                "Resistors, capacitors, sources, and other component identities remain correct.",
            ),
            _category(
                "labels_values_directions",
                "Labels, Values, and Directions",
                0.25,
                "Current directions, voltage labels, signs, and numeric values remain accurate.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.10,
                "The circuit is readable, uncluttered, and not missing major visible content.",
            ),
            _category(
                "background_color_fidelity",
                "Background Color Fidelity",
                0.10,
                "The canvas and panel backgrounds match the reference hue, brightness, tint, and any visible "
                "regional variation.",
            ),
        ],
        "cap_profile": BALANCED_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "circuit_semantic_error",
                2.5,
                "Wrong topology, wrong labels or values, or wrong current or voltage direction cap the score on this dataset.",
                any_of=[
                    {"id": "circuit_topology_connectivity", "max": 2.4},
                    {"id": "labels_values_directions", "max": 2.4},
                ],
            ),
        ],
    },
    "GEOQA_8K_R1V": {
        "dataset": "GEOQA_8K_R1V",
        "guidance": (
            "These are geometry-heavy figures. Be strict about structural precision, labels, markers, numbers, and "
            "whether each annotation clearly belongs to the right object."
        ),
        "prompt_notes": [
            "Slight label shifts are acceptable if the association is still unambiguous.",
            "Ambiguous or wrong label association should be penalized strongly.",
            "Precise geometry matters more than decorative drawing style.",
        ],
        "categories": [
            _category(
                "geometric_structure_precision",
                "Geometric Structure Precision",
                0.35,
                "Shapes, relations, and the core geometric construction remain precise and faithful.",
            ),
            _category(
                "numbers_labels_markers",
                "Numbers, Labels, and Markers",
                0.30,
                "Lengths, angles, labels, givens, and geometric markers remain accurate and readable.",
            ),
            _category(
                "association_placement_precision",
                "Association and Placement Precision",
                0.20,
                "Annotations are placed clearly enough that their target edge, angle, or point is unambiguous.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The figure is legible, unclipped, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "geometric_relation_or_association_error",
                2.5,
                "Wrong geometric relations or ambiguous label association cap the score on this dataset.",
                any_of=[
                    {"id": "geometric_structure_precision", "max": 2.4},
                    {"id": "association_placement_precision", "max": 2.4},
                ],
            ),
        ],
    },
    "Geoperception": {
        "dataset": "Geoperception",
        "guidance": (
            "These are geometry-heavy figures. Be strict about structure, labels, markers, and whether annotations clearly "
            "belong to the correct object or angle."
        ),
        "prompt_notes": [
            "Small label shifts are acceptable only if the target remains unambiguous.",
            "Wrong or ambiguous geometric association should be penalized strongly.",
            "Precise relation fidelity matters more than decorative style.",
        ],
        "categories": [
            _category(
                "geometric_structure_precision",
                "Geometric Structure Precision",
                0.35,
                "Shapes, intersections, and the core geometric relations remain precise and faithful.",
            ),
            _category(
                "numbers_labels_markers",
                "Numbers, Labels, and Markers",
                0.30,
                "Point labels, givens, numeric annotations, and geometric markers remain accurate and readable.",
            ),
            _category(
                "association_placement_precision",
                "Association and Placement Precision",
                0.20,
                "Annotations are positioned clearly enough that their intended target is unambiguous.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The figure is legible, unclipped, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "geometric_relation_or_association_error",
                2.5,
                "Wrong geometric relations or ambiguous label association cap the score on this dataset.",
                any_of=[
                    {"id": "geometric_structure_precision", "max": 2.4},
                    {"id": "association_placement_precision", "max": 2.4},
                ],
            ),
        ],
    },
    "geometry3k": {
        "dataset": "geometry3k",
        "guidance": (
            "These are geometry problem diagrams. Be strict about shape, relation fidelity, markers, labels, and whether "
            "annotations clearly belong to the correct geometric object."
        ),
        "prompt_notes": [
            "Do not give high scores to diagrams that are only approximately correct.",
            "Angle or length text can move slightly, but only if the association remains obvious.",
            "Missing or ambiguous markers should be penalized strongly.",
        ],
        "categories": [
            _category(
                "geometric_structure_precision",
                "Geometric Structure Precision",
                0.35,
                "Shapes, relations, and the core geometric construction remain precise and faithful.",
            ),
            _category(
                "numbers_labels_markers",
                "Numbers, Labels, and Markers",
                0.30,
                "Lengths, angles, point labels, and geometric markers remain accurate and readable.",
            ),
            _category(
                "association_placement_precision",
                "Association and Placement Precision",
                0.20,
                "Annotations are placed clearly enough that their target edge, angle, or point is unambiguous.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The figure is legible, unclipped, and not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "geometric_relation_or_association_error",
                2.5,
                "Wrong geometric relations or ambiguous label association cap the score on this dataset.",
                any_of=[
                    {"id": "geometric_structure_precision", "max": 2.4},
                    {"id": "association_placement_precision", "max": 2.4},
                ],
            ),
        ],
    },
    "Graph-Algorithms": {
        "dataset": "Graph-Algorithms",
        "guidance": (
            "Be strict about graph topology, directions, ordering, and labels. These diagrams can be dense, so stay fair "
            "when the source itself is messy, but do not forgive wrong graph structure."
        ),
        "prompt_notes": [
            "Exact topology and direction semantics matter more than cosmetic layout differences.",
            "Treat wrong ordering, weights, or labels as important errors.",
            "If the source graph is itself very cluttered, avoid over-penalizing ambiguity that already exists in the reference.",
        ],
        "categories": [
            _category(
                "graph_structure_fidelity",
                "Graph Structure Fidelity",
                0.40,
                "Nodes, edges, and overall graph topology remain correct.",
            ),
            _category(
                "ordering_direction_semantics",
                "Ordering and Direction Semantics",
                0.25,
                "Directions, ordering, and weights or edge semantics remain faithful.",
            ),
            _category(
                "label_number_precision",
                "Label and Number Precision",
                0.20,
                "Node ids, edge labels, and numeric annotations remain accurate and readable.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The graph remains readable and is not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "graph_topology_or_direction_error",
                2.5,
                "Wrong graph topology, ordering, or direction semantics cap the score on this dataset.",
                any_of=[
                    {"id": "graph_structure_fidelity", "max": 2.4},
                    {"id": "ordering_direction_semantics", "max": 2.4},
                ],
            ),
        ],
    },
    "GraphVQA-Swift": {
        "dataset": "GraphVQA-Swift",
        "guidance": (
            "Be strict about graph structure, label precision, and correct attribute-to-node or attribute-to-edge binding. "
            "Treat wrong label ordering or wrong attribute attachment as important errors."
        ),
        "prompt_notes": [
            "Topology and label precision matter more than cosmetic layout differences.",
            "Attributes such as colors or other markers must remain attached to the correct graph elements.",
            "If the source is visually messy, be fair about ambiguity already present in the reference.",
        ],
        "categories": [
            _category(
                "graph_structure_fidelity",
                "Graph Structure Fidelity",
                0.35,
                "Node-edge structure and the overall graph topology remain faithful.",
            ),
            _category(
                "label_number_precision",
                "Label and Number Precision",
                0.30,
                "Labels, counts, and numeric annotations remain accurate and attached to the correct elements.",
            ),
            _category(
                "attribute_mapping_fidelity",
                "Attribute Mapping Fidelity",
                0.20,
                "Colors or other attributes remain attached to the correct nodes or edges.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The graph remains readable and is not missing major visible content.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "graph_label_or_attribute_error",
                2.5,
                "Wrong label ordering or wrong attribute binding cap the score on this dataset.",
                any_of=[
                    {"id": "label_number_precision", "max": 2.4},
                    {"id": "attribute_mapping_fidelity", "max": 2.4},
                ],
            ),
        ],
    },
    "matplotlib": {
        "dataset": "matplotlib",
        "guidance": (
            "This is a coding-heavy recreation dataset with ground-truth code behind the reference. Be strict about layout, "
            "plot elements, ticks, annotations, and rendering fidelity. Unknown library-specific styling should not cause "
            "over-penalization when the core figure is otherwise correct."
        ),
        "prompt_notes": [
            "This dataset should generally score high only when the recreation is genuinely close.",
            "Wrong layout or missing major plot elements are important errors.",
            "Pure style misses from unknown libraries matter less than missing structure or annotations.",
        ],
        "categories": [
            _category(
                "layout_composition_fidelity",
                "Layout and Composition Fidelity",
                0.30,
                "Axes, subplot layout, framing, and composition match the reference.",
            ),
            _category(
                "plot_element_fidelity",
                "Plot Element Fidelity",
                0.25,
                "Lines, bars, markers, patches, and other major plotted elements are faithful.",
            ),
            _category(
                "text_ticks_annotations",
                "Text, Ticks, and Annotations",
                0.25,
                "Titles, labels, tick text, legends, and annotations are present, accurate, and well placed.",
            ),
            _category(
                "style_rendering_fidelity",
                "Style and Rendering Fidelity",
                0.20,
                "Line styles, fonts, colors, and overall rendering feel resemble the reference.",
            ),
        ],
        "cap_profile": STRICT_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "major_plot_layout_failure",
                3.0,
                "Wrong layout or missing major plot elements cap the score on this dataset.",
                any_of=[
                    {"id": "layout_composition_fidelity", "max": 2.4},
                    {"id": "plot_element_fidelity", "max": 2.4},
                ],
            ),
        ],
    },
    "OlympiadBench": {
        "dataset": "OlympiadBench",
        "guidance": (
            "These figures are harder and often denser than basic geometry. Be strict about the problem-relevant structure, "
            "labels, givens, and symbols, but stay fair when the source itself is complex or visually noisy."
        ),
        "prompt_notes": [
            "Judge whether the recreated figure would support the same downstream reasoning problem.",
            "Wrong core problem structure matters more than decorative style.",
            "Because these are harder figures, do not over-penalize small cosmetic differences that do not change meaning.",
        ],
        "categories": [
            _category(
                "problem_structure_semantics",
                "Problem Structure and Semantics",
                0.35,
                "The core problem diagram and its essential relations remain faithful.",
            ),
            _category(
                "labels_numbers_symbols",
                "Labels, Numbers, and Symbols",
                0.25,
                "Given values, labels, and symbols remain accurate and readable.",
            ),
            _category(
                "spatial_precision",
                "Spatial Precision",
                0.20,
                "Placement and construction remain faithful enough for the same reasoning task.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.20,
                "The diagram remains readable, unclipped, and complete enough to use.",
            ),
        ],
        "cap_profile": BALANCED_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "wrong_core_problem_semantics",
                2.5,
                "Wrong core problem semantics cap the score on this dataset.",
                any_of=[
                    {"id": "problem_structure_semantics", "max": 2.4},
                ],
            ),
        ],
    },
    "Physics": {
        "dataset": "Physics",
        "guidance": (
            "Focus on physical setup, object placement, vectors, labels, and numerical givens. Wrong force direction or "
            "physically inconsistent placement should be penalized strongly."
        ),
        "prompt_notes": [
            "Check vector direction, labels, and scene consistency carefully.",
            "The object should sit correctly in the depicted setup, for example on a slope rather than floating away from it.",
            "Minor cosmetic style differences matter less than physical correctness.",
        ],
        "categories": [
            _category(
                "physical_setup_fidelity",
                "Physical Setup Fidelity",
                0.35,
                "The overall scene and essential physical setup remain faithful.",
            ),
            _category(
                "vectors_labels_numbers",
                "Vectors, Labels, and Numbers",
                0.30,
                "Force arrows, directions, angles, labels, and values remain accurate and readable.",
            ),
            _category(
                "object_geometry_consistency",
                "Object Geometry Consistency",
                0.20,
                "Object placement and geometry remain physically consistent with the reference.",
            ),
            _category(
                "readability_completeness",
                "Readability and Completeness",
                0.15,
                "The diagram is readable, unclipped, and not missing major visible content.",
            ),
        ],
        "cap_profile": BALANCED_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "vector_or_physical_consistency_error",
                2.5,
                "Wrong vector direction or physically inconsistent placement cap the score on this dataset.",
                any_of=[
                    {"id": "vectors_labels_numbers", "max": 2.4},
                    {"id": "object_geometry_consistency", "max": 2.4},
                ],
            ),
        ],
    },
    "spatialvlm_qa": {
        "dataset": "spatialvlm_qa",
        "guidance": (
            "This is a hard 3D spatial reconstruction dataset. Judge object identity, spatial relations, depth, and scene "
            "completion carefully. Stay fair on difficulty, but do not ignore wrong core spatial relations."
        ),
        "prompt_notes": [
            "Correct objects but a flattened 2D recreation should land around the low-to-mid range rather than score highly.",
            "Wrong core spatial relation such as front versus behind is a major error.",
            "Depth, volume, and 3D cues matter more here than on flat diagram datasets.",
        ],
        "categories": [
            _category(
                "object_shape_identity_fidelity",
                "Object Shape and Identity Fidelity",
                0.25,
                "The right objects, basic shapes, and counts are present.",
            ),
            _category(
                "spatial_relation_correctness",
                "Spatial Relation Correctness",
                0.35,
                "Relative positions such as left or right, above or below, and front or behind are correct.",
            ),
            _category(
                "three_d_depth_rendering",
                "3D Depth Rendering",
                0.25,
                "Depth, volume, shadows, and 3D reconstruction cues are preserved.",
            ),
            _category(
                "color_scene_completeness",
                "Color and Scene Completeness",
                0.15,
                "The scene remains reasonably complete, with major colors and scene elements preserved.",
            ),
        ],
        "cap_profile": HARD_CAP_PROFILE,
        "extra_caps": [
            _extra_cap(
                "flattened_two_d_scene",
                2.5,
                "A scene that preserves objects but collapses the reference into a flat 2D depiction cannot score highly.",
                all_of=[
                    {"id": "object_shape_identity_fidelity", "min": 3.0},
                    {"id": "three_d_depth_rendering", "max": 2.4},
                ],
            ),
            _extra_cap(
                "wrong_core_spatial_relation",
                2.0,
                "Wrong core spatial relations cap the score on this dataset.",
                any_of=[
                    {"id": "spatial_relation_correctness", "max": 2.0},
                ],
            ),
        ],
    },
}

# Separate opt-in category; never mutate the legacy chemistry definition.
CHEMISTRY_SPACING_RUBRIC = "ChemVQA-2K-spacing-v1"
_chemistry_spacing = deepcopy(DATASET_RUBRICS["ChemVQA-2K"])
_chemistry_spacing["dataset"] = CHEMISTRY_SPACING_RUBRIC
_chemistry_spacing["guidance"] += (
    " Correct chemistry alone is insufficient for 5: local label and bond clearance must also match the reference."
)
_chemistry_spacing["prompt_notes"] += [
    "Inspect the tightest atom-symbol/bond junction, atom-index placement, and charge/subscript group. "
    "In the readability rationale, name the tightest location and compare its clearance with the reference.",
    "New crowding includes bonds touching or crossing glyphs, indices touching bonds or atom symbols, "
    "and charges/subscripts merging with neighboring labels. Penalize added crowding even if the chemistry is inferable.",
    "Readability anchors: 5 = no added crowding; 4.5-4.9 = slightly reduced but clear separation; "
    "at most 4 = clear new crowding or contact; at most 3 = overlap obscures a symbol, charge, index, or bond order.",
    "Do not penalize density already present in the reference, harmless font/scale changes, or intended bond junctions. "
    "Lower symbol or bond scores too only when their own accuracy or legibility is affected.",
    "Any reported spacing defect must lower readability below 5. Clear new crowding caps the final score at 4.5; "
    "obscuring overlap caps it at 4.0. Any category below 5 prevents a rounded perfect score.",
]
_chemistry_spacing["categories"][-1]["description"] = (
    "The structure is complete and unclipped; atom symbols, indices, charges, subscripts, and adjacent bonds "
    "retain clear separation relative to the reference."
)
_chemistry_spacing["extra_caps"] += [
    _extra_cap(
        "chemistry_no_rounded_perfect",
        4.9,
        "A category below 5 cannot round to a perfect chemistry rating.",
        any_of=[{"id": category["id"], "max": 4.9} for category in _chemistry_spacing["categories"]],
    ),
    _extra_cap(
        "chemistry_added_crowding",
        4.5,
        "Clear added crowding prevents a near-perfect chemistry rating.",
        any_of=[{"id": "readability_completeness", "max": 4.0}],
    ),
    _extra_cap(
        "chemistry_obscuring_overlap",
        4.0,
        "Overlap obscuring notation limits the chemistry rating.",
        any_of=[{"id": "readability_completeness", "max": 3.0}],
    ),
]
DATASET_RUBRICS[CHEMISTRY_SPACING_RUBRIC] = _chemistry_spacing

RATER_SYSTEM_PROMPT = """\
You are a rigorous but calibrated evaluator for image recreation quality.
Compare the reference image against the candidate rendered image.
Judge only what is visibly present in the images.

Use the dataset-specific rubric and instructions provided by the user message.
Be strict about semantically wrong recreations, but do not over-penalize differences that the rubric explicitly says
to treat as secondary.

Scoring anchors on a 0.0 to 5.0 scale:
- 5.0 = near-exact or clearly excellent for that dataset
- 4.0 = strong recreation with minor issues
- 3.0 = good but with noticeable structural, text, or semantic problems
- 2.0 = partial recreation with important errors
- 1.0 = major mismatch
- 0.0 = missing, unreadable, blank, or fundamentally broken

Return JSON only. Do not include markdown fences.
"""


def encode_image_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    # Preparation accepts images by their contents, including extensionless files.
    with Image.open(image_path) as image:
        mime = Image.MIME[image.format]
    return f"data:{mime};base64,{b64}"


def _comparison_content(
    *, source_image_path: Path, rendered_image_path: Path, prompt_text: str
) -> List[Dict[str, Any]]:
    return [
        {"type": "text", "text": "Reference image:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(source_image_path)}},
        {"type": "text", "text": "Candidate rendered image:"},
        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(rendered_image_path)}},
        {"type": "text", "text": prompt_text},
    ]


def rubric_markdown(rubric: Mapping[str, Any]) -> str:
    lines = [f"Dataset: {rubric['dataset']}"]
    for category in rubric["categories"]:
        lines.append(f"- {category['id']} (weight {category['weight']:.2f}): {category['description']}")
    return "\n".join(lines)


def build_rater_user_content(
    *,
    source_image_path: Path,
    rendered_image_path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """Build the dataset judge prompt with reference and candidate images."""
    rubric = DATASET_RUBRICS[dataset_name]
    metadata_payload = dict(metadata or {})
    prompt_notes = "\n".join(f"- {note}" for note in rubric.get("prompt_notes", []))
    revision_note = f"Dataset rubric revision: {rubric['revision']}\n" if "revision" in rubric else ""
    prompt_text = (
        "Evaluate the candidate recreation against the reference image using the dataset-specific rubric below.\n\n"
        f"Rubric version: {RUBRIC_VERSION}\n"
        f"{revision_note}"
        f"Rubric dataset: {dataset_name}\n"
        f"Scoring guidance: {rubric['guidance']}\n\n"
        "Dataset-specific notes:\n"
        f"{prompt_notes}\n\n"
        "Categories:\n"
        f"{rubric_markdown(rubric)}\n\n"
        "Reference metadata:\n"
        f"{json.dumps(metadata_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        '1) "category_scores"\n'
        '2) "rationales"\n'
        '3) "strengths"\n'
        '4) "issues"\n'
        '5) "overall_summary"\n\n'
        "Rules:\n"
        "- 'category_scores' must contain every category id with a numeric score from 0.0 to 5.0.\n"
        "- Use decimals in increments of 0.1 when needed.\n"
        "- 'rationales' must contain the same keys, with one short rationale per category.\n"
        "- 'strengths' must be a short list of 1 to 3 bullets.\n"
        "- 'issues' must be a short list of 1 to 5 bullets.\n"
        "- 'overall_summary' must be 1 to 3 concise sentences.\n"
        "- Judge what matters for this dataset-specific rubric, not superficial similarity alone.\n"
        "- For easy datasets, preserve recreation structure, not just semantic content.\n"
        "- For hard or noisy references, be strict but do not penalize differences that the source itself makes ambiguous.\n"
        "- Use 0.0 or 1.0 for severe failures.\n"
        "- Do not compute the final 0.0 to 5.0 rating yourself; provide category scores only.\n"
    )

    return _comparison_content(
        source_image_path=source_image_path, rendered_image_path=rendered_image_path, prompt_text=prompt_text
    )


def strip_thinking_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<think>"):
        stripped = re.sub(r"^\s*<think>.*?</think>\s*", "", stripped, count=1, flags=re.DOTALL)
    return stripped.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_thinking_content(text)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj

    raise ValueError("Could not parse response as a JSON object.")


def _round_tenth(value: float) -> float:
    return round(float(value) + 1e-8, 1)


def _coerce_decimal_score(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean score for {key}.")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str) and value.strip():
        score = float(value.strip())
    else:
        raise ValueError(f"Missing score for {key}.")

    if not math.isfinite(score):
        raise ValueError(f"Invalid score for {key}: {value!r}")
    if score < SCORE_MIN or score > SCORE_MAX:
        raise ValueError(f"Score for {key} must be between {SCORE_MIN:.1f} and {SCORE_MAX:.1f} (got {score}).")
    return _round_tenth(score)


def inspect_candidate_image(image_path: Path) -> Dict[str, Any]:
    if not image_path.exists():
        return {
            "candidate_readable": False,
            "is_blank_or_degenerate": True,
            "checks_triggered": ["missing_candidate_image"],
            "width": 0,
            "height": 0,
        }

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            gray = rgb.convert("L")
            stat = ImageStat.Stat(gray)
            mean_value = float(stat.mean[0]) if stat.mean else 0.0
            variance = float(stat.var[0]) if stat.var else 0.0
            extrema = gray.getextrema() or (0, 0)
            histogram = gray.histogram()
            total_pixels = max(width * height, 1)
            near_white = sum(histogram[250:256]) / total_pixels
            near_black = sum(histogram[:6]) / total_pixels
            checks: List[str] = []
            if min(width, height) < 32:
                checks.append("too_small")
            if variance < 4.0:
                checks.append("near_zero_variance")
            if extrema[1] - extrema[0] < 10:
                checks.append("near_monochrome")
            if near_white > 0.985 or near_black > 0.985:
                checks.append("mostly_empty_canvas")
            scoring_checks = [check for check in checks if check != "mostly_empty_canvas"]
            return {
                "candidate_readable": True,
                # Whitespace alone is diagnostic; sparse diagrams can be valid.
                "is_blank_or_degenerate": bool(scoring_checks),
                "checks_triggered": checks,
                "width": width,
                "height": height,
                "grayscale_mean": round(mean_value, 4),
                "grayscale_variance": round(variance, 4),
                "near_white_ratio": round(near_white, 6),
                "near_black_ratio": round(near_black, 6),
            }
    except Exception as exc:
        return {
            "candidate_readable": False,
            "is_blank_or_degenerate": True,
            "checks_triggered": [f"unreadable_image:{type(exc).__name__}"],
            "width": 0,
            "height": 0,
        }


def final_rating_from_raw_v2(raw_score_0_to_5: float) -> float:
    clamped = max(SCORE_MIN, min(SCORE_MAX, float(raw_score_0_to_5)))
    return _round_tenth(clamped)


def _critical_categories(rubric: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return rubric["categories"][:2]


def _condition_matches(condition: Mapping[str, float], category_results: Mapping[str, Mapping[str, Any]]) -> bool:
    category_id = str(condition.get("id") or "")
    if category_id not in category_results:
        return False
    score = float(category_results[category_id]["score_0_to_5"])
    if "min" in condition and score < float(condition["min"]) - 1e-8:
        return False
    if "max" in condition and score > float(condition["max"]) + 1e-8:
        return False
    return True


def apply_rating_caps(
    *,
    provisional_rating_0_to_5: float,
    rubric: Mapping[str, Any],
    category_results: Mapping[str, Mapping[str, Any]],
    execution_status: str,
    candidate_inspection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    final_rating = float(provisional_rating_0_to_5)
    applied_caps: List[Dict[str, Any]] = []

    def apply_cap(cap_to: float, reason: str, cap_type: str) -> None:
        nonlocal final_rating
        previous = final_rating
        final_rating = min(final_rating, float(cap_to))
        applied_caps.append(
            {
                "type": cap_type,
                "cap_to": _round_tenth(cap_to),
                "reason": reason,
                "changed_final_rating": final_rating < previous - 1e-8,
            }
        )

    if execution_status != "ok":
        apply_cap(0.0, f"Execution did not produce a valid render: {execution_status}", "execution_failure")

    candidate_readable = bool((candidate_inspection or {}).get("candidate_readable", False))
    if execution_status == "ok" and not candidate_readable:
        apply_cap(0.0, "Candidate image is missing or unreadable.", "candidate_missing_or_unreadable")

    if (candidate_inspection or {}).get("is_blank_or_degenerate"):
        triggered_checks = list((candidate_inspection or {}).get("checks_triggered", []))
        # Sparse diagrams can validly occupy little canvas; this check is diagnostic only.
        scoring_checks = [check for check in triggered_checks if check != "mostly_empty_canvas"]
        # A caller-provided blank flag without detailed checks remains a scoring failure.
        if scoring_checks or not triggered_checks:
            checks = ", ".join(scoring_checks) or "blank_or_degenerate"
            apply_cap(0.5, f"Candidate image appears blank or degenerate: {checks}", "blank_or_degenerate")

    cap_profile = rubric.get("cap_profile", BALANCED_CAP_PROFILE)
    criticals = list(_critical_categories(rubric))
    critical_scores = [float(category_results[category["id"]]["score_0_to_5"]) for category in criticals]

    if any(score <= float(cap_profile["critical_any_threshold"]) for score in critical_scores):
        apply_cap(
            float(cap_profile["critical_any_cap"]),
            "At least one critical category received a very low score.",
            "critical_category_floor",
        )

    if len(critical_scores) >= 2 and all(
        score <= float(cap_profile["critical_pair_threshold"]) for score in critical_scores[:2]
    ):
        apply_cap(
            float(cap_profile["critical_pair_cap"]),
            "Both critical categories are weak enough to limit the final score.",
            "multiple_critical_weaknesses",
        )

    if provisional_rating_0_to_5 >= float(cap_profile["top_score_trigger"]) and any(
        score < float(cap_profile["top_score_min_critical"]) for score in critical_scores
    ):
        apply_cap(
            float(cap_profile["top_score_cap"]),
            "A near-perfect final score requires both critical categories to be very strong.",
            "top_score_guardrail",
        )

    for extra_cap in rubric.get("extra_caps", []):
        all_conditions = extra_cap.get("all_of") or []
        any_conditions = extra_cap.get("any_of") or []
        all_met = all(_condition_matches(condition, category_results) for condition in all_conditions)
        any_met = (
            True
            if not any_conditions
            else any(_condition_matches(condition, category_results) for condition in any_conditions)
        )
        if all_met and any_met:
            apply_cap(float(extra_cap["cap_to"]), str(extra_cap["reason"]), str(extra_cap["type"]))

    final_rating = max(SCORE_MIN, min(SCORE_MAX, final_rating))
    return {
        "final_rating_0_to_5": _round_tenth(final_rating),
        "applied_caps": applied_caps,
    }


def _score_categories(
    parsed: Dict[str, Any], *, categories: Sequence[Mapping[str, Any]], critical_count: int
) -> tuple[Dict[str, Any], float]:
    """Validate judge scores and accumulate their weighted category results."""
    scores_obj = parsed.get("category_scores")
    if not isinstance(scores_obj, dict):
        scores_obj = parsed.get("subscores")
    rationales_obj = parsed.get("rationales")
    if not isinstance(scores_obj, dict):
        raise ValueError("Response is missing a 'category_scores' object.")
    if not isinstance(rationales_obj, dict):
        raise ValueError("Response is missing a 'rationales' object.")

    category_results: Dict[str, Any] = {}
    raw_score = 0.0
    for idx, category in enumerate(categories):
        category_id = category["id"]
        score = _coerce_decimal_score(scores_obj.get(category_id), category_id)
        weighted = score * float(category["weight"])
        raw_score += weighted
        category_results[category_id] = {
            "label": category["label"],
            "description": category["description"],
            "weight": category["weight"],
            "score_0_to_5": score,
            "weighted_score_0_to_5": round(weighted, 4),
            "rationale": str(rationales_obj.get(category_id, "")).strip(),
            "is_critical": idx < critical_count,
        }
    return category_results, raw_score


def aggregate_rating_v2(
    parsed: Dict[str, Any],
    *,
    dataset_name: str,
    execution_status: str = "ok",
    execution_error_present: bool = False,
    candidate_inspection: Optional[Mapping[str, Any]] = None,
    reference_image_path: str = "",
    candidate_image_path: str = "",
    rubric_version: str = RUBRIC_VERSION,
) -> Dict[str, Any]:
    """Aggregate dataset scores with critical-category and image-degeneracy caps."""
    rubric = DATASET_RUBRICS[dataset_name]
    category_results, raw_score = _score_categories(parsed, categories=rubric["categories"], critical_count=2)

    strengths = [str(item).strip() for item in parsed.get("strengths", []) if str(item).strip()]
    issues = [str(item).strip() for item in parsed.get("issues", []) if str(item).strip()]
    overall_summary = str(parsed.get("overall_summary", "")).strip()

    provisional = final_rating_from_raw_v2(raw_score)
    cap_result = apply_rating_caps(
        provisional_rating_0_to_5=raw_score,
        rubric=rubric,
        category_results=category_results,
        execution_status=execution_status,
        candidate_inspection=candidate_inspection,
    )

    if execution_status != "ok":
        issues = [f"Render failed before visual evaluation: {execution_status}"]
        overall_summary = (
            "The generated code did not produce a valid rendered image, so the sample receives the minimum rating."
        )

    return {
        "rubric_version": rubric_version,
        "rubric_dataset": dataset_name,
        **({"rubric_revision": rubric["revision"]} if "revision" in rubric else {}),
        "status": "ok" if execution_status == "ok" else "render_failed",
        "execution_status": execution_status,
        "execution_detail": "" if execution_status == "ok" else execution_status,
        "execution_error_present": execution_error_present,
        "reference_image_path": reference_image_path,
        "candidate_image_path": candidate_image_path,
        "candidate_inspection": dict(candidate_inspection or {}),
        "category_scores": category_results,
        "raw_score_0_to_5": round(raw_score, 4),
        "raw_visual_score_0_to_5": round(raw_score, 4),
        "provisional_rating_0_to_5": provisional,
        "applied_caps": cap_result["applied_caps"],
        "final_rating_0_to_5": cap_result["final_rating_0_to_5"],
        "strengths": strengths[:3],
        "issues": issues[:5],
        "overall_summary": overall_summary,
    }


def _render_failure_result(
    status: str,
    *,
    categories: Sequence[Mapping[str, Any]],
    critical_count: int,
    execution_error_present: bool = True,
    reference_image_path: str = "",
    candidate_image_path: str = "",
    candidate_inspection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    category_scores: Dict[str, Any] = {}
    for idx, category in enumerate(categories):
        category_scores[category["id"]] = {
            "label": category["label"],
            "description": category["description"],
            "weight": category["weight"],
            "score_0_to_5": 0.0,
            "weighted_score_0_to_5": 0.0,
            "rationale": f"Candidate image unavailable because generation or render failed: {status}",
            "is_critical": idx < critical_count,
        }

    return {
        "status": "render_failed",
        "execution_status": "render_failed",
        "execution_detail": status,
        "execution_error_present": execution_error_present,
        "reference_image_path": reference_image_path,
        "candidate_image_path": candidate_image_path,
        "candidate_inspection": dict(candidate_inspection or {}),
        "category_scores": category_scores,
        "raw_score_0_to_5": 0.0,
        "raw_visual_score_0_to_5": 0.0,
        "provisional_rating_0_to_5": 0.0,
        "final_rating_0_to_5": 0.0,
        "strengths": [],
        "issues": [f"Render failed before visual evaluation: {status}"],
        "overall_summary": "The generated code did not produce a valid rendered image, so the sample receives the minimum rating.",
    }


def build_render_failure_rating(
    status: str,
    *,
    dataset_name: str,
    execution_error_present: bool = True,
    reference_image_path: str = "",
    candidate_image_path: str = "",
    candidate_inspection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return zero category scores and an execution-failure cap."""
    rubric = DATASET_RUBRICS[dataset_name]
    return {
        **_render_failure_result(
            status,
            categories=rubric["categories"],
            critical_count=2,
            execution_error_present=execution_error_present,
            reference_image_path=reference_image_path,
            candidate_image_path=candidate_image_path,
            candidate_inspection=candidate_inspection,
        ),
        "rubric_version": RUBRIC_VERSION,
        "rubric_dataset": dataset_name,
        **({"rubric_revision": rubric["revision"]} if "revision" in rubric else {}),
        "applied_caps": [
            {
                "type": "execution_failure",
                "cap_to": 0.0,
                "reason": f"Execution did not produce a valid render: {status}",
                "changed_final_rating": True,
            }
        ],
    }
