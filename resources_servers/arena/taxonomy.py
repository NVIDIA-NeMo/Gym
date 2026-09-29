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
"""LMArena prompt taxonomy."""

INDUSTRIES = (
    "software_and_it_services",
    "writing_and_literature_and_language",
    "life_and_physical_and_social_science",
    "entertainment_and_sports_and_media",
    "business_and_management_and_financial_operations",
    "mathematical",
    "legal_and_government",
    "medicine_and_healthcare",
)
LANGUAGES = {
    "chinese": {"zh", "zh-Hant"},
    "french": {"fr"},
    "german": {"de"},
    "spanish": {"es"},
    "russian": {"ru"},
    "japanese": {"ja"},
    "korean": {"ko"},
    "polish": {"pl"},
}
PROMPT_CATEGORY_ORDER = (
    "expert",
    # Metadata uses underscore-separated industry keys; leaderboard categories use
    # the public "industry-..." names with hyphens.
    *("industry-" + name.replace("_", "-") for name in INDUSTRIES),
    "math",
    "instruction-following",
    "multi-turn",
    "creative-writing",
    "coding",
    "hard-prompts",
    "hard-prompts-english",
    "english",
    "non-english",
    *LANGUAGES,
)
MIN_SLICE_PROMPTS = 50


def _has_tag(tags: dict, value_tag: str, nested_tag: str, nested_field: str) -> bool:
    # Records use either {value_tag: {"value": bool}} or
    # {nested_tag: {nested_field: bool}}.
    return (tags.get(value_tag) or {}).get("value") is True or (tags.get(nested_tag) or {}).get(nested_field) is True


def get_prompt_categories(row: dict) -> set[str]:
    metadata = row.get("metadata") or {}
    tags = metadata.get("tags") or {}
    language = metadata.get("user_language")
    categories = set()

    for name, value_tag, nested_tag, nested_field in (
        ("expert", "expert_v1", "expert_v0.1", "expert"),
        ("math", "math_v1", "math_v0.1", "math"),
        ("instruction-following", "instruction_following_v1", "if_v0.1", "if"),
        ("creative-writing", "creative_writing_v1", "creative_writing_v0.1", "creative_writing"),
    ):
        if _has_tag(tags, value_tag, nested_tag, nested_field):
            categories.add(name)

    if _has_tag(tags, "coding_v2", "coding_v0.1", "coding") or metadata.get("is_code") is True:
        categories.add("coding")

    hard = tags.get("hard_prompt_v1") or tags.get("criteria_v0.1") or {}
    if sum(value is True for key, value in hard.items() if key != "tags") >= 6:
        categories.add("hard-prompts")
        if language == "en":
            categories.add("hard-prompts-english")

    inputs = row.get("responses_create_params", {}).get("input", [])
    if sum(message.get("role") == "user" for message in inputs) > 1:
        categories.add("multi-turn")

    for industry in INDUSTRIES:
        if (tags.get("industry_v1") or {}).get(industry) is True or (tags.get("industry_v0.1") or {}).get(
            industry
        ) is True:
            categories.add("industry-" + industry.replace("_", "-"))

    if language == "en":
        categories.add("english")
    elif language not in {None, "und", "<err>"}:
        categories.add("non-english")
    for name, codes in LANGUAGES.items():
        if language in codes:
            categories.add(name)
    return categories


def get_taxonomy_values(row: dict, field: str) -> set[str]:
    """Return the non-empty values of one taxonomy field for a prompt."""
    taxonomy = (row.get("metadata") or {}).get("taxonomy") or []
    return {item[field] for item in taxonomy if item.get(field) not in {None, "None"}}


def get_prompt_slices(row: dict) -> dict[str, set[str]]:
    """Return the three overlapping prompt taxonomies reported with scores."""
    return {
        "arena": get_prompt_categories(row),
        "taxonomy-language": get_taxonomy_values(row, "natural_language"),
        "taxonomy-task-type": get_taxonomy_values(row, "task_type"),
    }
