# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""BambooHR REST API routing layer for api_fetch.

Dispatches incoming REST-style BambooHR requests to their matching implementation
functions. Modeled after the BambooHR REST API (https://documentation.bamboohr.com).
"""

from automationbench.tools.api.impl.bamboohr import (
    bamboohr_employees_changed,
    bamboohr_employees_create,
    bamboohr_employees_directory,
    bamboohr_employees_files_delete,
    bamboohr_employees_files_get,
    bamboohr_employees_files_list,
    bamboohr_employees_files_update,
    bamboohr_employees_files_upload,
    bamboohr_employees_get,
    bamboohr_employees_tables_add_row,
    bamboohr_employees_tables_changed,
    bamboohr_employees_tables_get,
    bamboohr_employees_tables_update_row,
    bamboohr_employees_update,
    bamboohr_goals_aggregate,
    bamboohr_goals_create,
    bamboohr_goals_delete,
    bamboohr_goals_list,
    bamboohr_goals_status_count,
    bamboohr_goals_update,
    bamboohr_reports_company,
    bamboohr_reports_custom,
    bamboohr_timeoff_calculator,
    bamboohr_timeoff_history_add,
    bamboohr_timeoff_policies_assign,
    bamboohr_timeoff_policies_list,
    bamboohr_timeoff_requests_change_status,
    bamboohr_timeoff_requests_create,
    bamboohr_timeoff_requests_list,
    bamboohr_timeoff_types_list,
    bamboohr_timeoff_whos_out,
    bamboohr_training_categories_create,
    bamboohr_training_categories_delete,
    bamboohr_training_categories_list,
    bamboohr_training_categories_update,
    bamboohr_training_records_create,
    bamboohr_training_records_delete,
    bamboohr_training_records_employee,
    bamboohr_training_records_update,
    bamboohr_training_types_create,
    bamboohr_training_types_delete,
    bamboohr_training_types_list,
    bamboohr_training_types_update,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Employees
    ("GET", r"bamboohr/v1/employees/directory$", "employees_directory"),
    ("GET", r"bamboohr/v1/employees/changed$", "employees_changed"),
    ("POST", r"bamboohr/v1/employees$", "employees_create"),
    ("GET", r"bamboohr/v1/employees/([^/]+)$", "employees_get"),
    ("POST", r"bamboohr/v1/employees/([^/]+)$", "employees_update"),
    # Employee Files
    ("GET", r"bamboohr/v1/employees/([^/]+)/files/view$", "employees_files_list"),
    ("POST", r"bamboohr/v1/employees/([^/]+)/files$", "employees_files_upload"),
    ("GET", r"bamboohr/v1/employees/([^/]+)/files/([^/]+)$", "employees_files_get"),
    ("POST", r"bamboohr/v1/employees/([^/]+)/files/([^/]+)$", "employees_files_update"),
    ("DELETE", r"bamboohr/v1/employees/([^/]+)/files/([^/]+)$", "employees_files_delete"),
    # Employee Tables
    ("GET", r"bamboohr/v1/employees/changed/tables/([^/]+)$", "employees_tables_changed"),
    ("GET", r"bamboohr/v1/employees/([^/]+)/tables/([^/]+)$", "employees_tables_get"),
    ("POST", r"bamboohr/v1/employees/([^/]+)/tables/([^/]+)$", "employees_tables_add_row"),
    (
        "POST",
        r"bamboohr/v1/employees/([^/]+)/tables/([^/]+)/([^/]+)$",
        "employees_tables_update_row",
    ),
    # Time Off
    ("GET", r"bamboohr/v1/time_off/requests$", "timeoff_requests_list"),
    ("PUT", r"bamboohr/v1/employees/([^/]+)/time_off/request$", "timeoff_requests_create"),
    ("PUT", r"bamboohr/v1/time_off/requests/([^/]+)/status$", "timeoff_requests_change_status"),
    ("PUT", r"bamboohr/v1/employees/([^/]+)/time_off/history$", "timeoff_history_add"),
    ("GET", r"bamboohr/v1/time_off/whos_out$", "timeoff_whos_out"),
    ("GET", r"bamboohr/v1/employees/([^/]+)/time_off/calculator$", "timeoff_calculator"),
    ("GET", r"bamboohr/v1/meta/time_off/policies$", "timeoff_policies_list"),
    ("GET", r"bamboohr/v1/meta/time_off/types$", "timeoff_types_list"),
    ("PUT", r"bamboohr/v1/employees/([^/]+)/time_off/policies$", "timeoff_policies_assign"),
    # Reports
    ("GET", r"bamboohr/v1/reports/([^/]+)$", "reports_company"),
    ("POST", r"bamboohr/v1/reports/custom$", "reports_custom"),
    # Goals (Performance)
    ("GET", r"bamboohr/v1/performance/employees/([^/]+)/goals/statusCount$", "goals_status_count"),
    ("GET", r"bamboohr/v1/performance/employees/([^/]+)/goals/aggregate$", "goals_aggregate"),
    ("GET", r"bamboohr/v1/performance/employees/([^/]+)/goals$", "goals_list"),
    ("POST", r"bamboohr/v1/performance/employees/([^/]+)/goals$", "goals_create"),
    ("PUT", r"bamboohr/v1/performance/employees/([^/]+)/goals/([^/]+)$", "goals_update"),
    ("DELETE", r"bamboohr/v1/performance/employees/([^/]+)/goals/([^/]+)$", "goals_delete"),
    # Training
    ("GET", r"bamboohr/v1/training/type$", "training_types_list"),
    ("POST", r"bamboohr/v1/training/type$", "training_types_create"),
    ("PUT", r"bamboohr/v1/training/type/([^/]+)$", "training_types_update"),
    ("DELETE", r"bamboohr/v1/training/type/([^/]+)$", "training_types_delete"),
    ("GET", r"bamboohr/v1/training/category$", "training_categories_list"),
    ("POST", r"bamboohr/v1/training/category$", "training_categories_create"),
    ("PUT", r"bamboohr/v1/training/category/([^/]+)$", "training_categories_update"),
    ("DELETE", r"bamboohr/v1/training/category/([^/]+)$", "training_categories_delete"),
    ("GET", r"bamboohr/v1/training/record/([^/]+)$", "training_records_employee"),
    ("POST", r"bamboohr/v1/training/record/([^/]+)$", "training_records_create"),
    ("PUT", r"bamboohr/v1/training/record/([^/]+)/([^/]+)$", "training_records_update"),
    ("DELETE", r"bamboohr/v1/training/record/([^/]+)/([^/]+)$", "training_records_delete"),
]

_HANDLERS = {
    # Employees
    "employees_directory": lambda w, ids, p, b: bamboohr_employees_directory(w, **p),
    "employees_changed": lambda w, ids, p, b: bamboohr_employees_changed(w, **p),
    "employees_create": lambda w, ids, p, b: bamboohr_employees_create(w, **b),
    "employees_get": lambda w, ids, p, b: bamboohr_employees_get(w, employee_id=ids[0], **p),
    "employees_update": lambda w, ids, p, b: bamboohr_employees_update(w, employee_id=ids[0], **b),
    # Employee Files
    "employees_files_list": lambda w, ids, p, b: bamboohr_employees_files_list(
        w, employee_id=ids[0], **p
    ),
    "employees_files_upload": lambda w, ids, p, b: bamboohr_employees_files_upload(
        w, employee_id=ids[0], **b
    ),
    "employees_files_get": lambda w, ids, p, b: bamboohr_employees_files_get(
        w, employee_id=ids[0], file_id=ids[1], **p
    ),
    "employees_files_update": lambda w, ids, p, b: bamboohr_employees_files_update(
        w, employee_id=ids[0], file_id=ids[1], **b
    ),
    "employees_files_delete": lambda w, ids, p, b: bamboohr_employees_files_delete(
        w, employee_id=ids[0], file_id=ids[1]
    ),
    # Employee Tables
    "employees_tables_changed": lambda w, ids, p, b: bamboohr_employees_tables_changed(
        w, table=ids[0], **p
    ),
    "employees_tables_get": lambda w, ids, p, b: bamboohr_employees_tables_get(
        w, employee_id=ids[0], table=ids[1], **p
    ),
    "employees_tables_add_row": lambda w, ids, p, b: bamboohr_employees_tables_add_row(
        w, employee_id=ids[0], table=ids[1], **b
    ),
    "employees_tables_update_row": lambda w, ids, p, b: bamboohr_employees_tables_update_row(
        w, employee_id=ids[0], table=ids[1], row_id=ids[2], **b
    ),
    # Time Off
    "timeoff_requests_list": lambda w, ids, p, b: bamboohr_timeoff_requests_list(w, **p),
    "timeoff_requests_create": lambda w, ids, p, b: bamboohr_timeoff_requests_create(
        w, employee_id=ids[0], **b
    ),
    "timeoff_requests_change_status": lambda w, ids, p, b: bamboohr_timeoff_requests_change_status(
        w, request_id=ids[0], **b
    ),
    "timeoff_history_add": lambda w, ids, p, b: bamboohr_timeoff_history_add(
        w, employee_id=ids[0], **b
    ),
    "timeoff_whos_out": lambda w, ids, p, b: bamboohr_timeoff_whos_out(w, **p),
    "timeoff_calculator": lambda w, ids, p, b: bamboohr_timeoff_calculator(
        w, employee_id=ids[0], **p
    ),
    "timeoff_policies_list": lambda w, ids, p, b: bamboohr_timeoff_policies_list(w),
    "timeoff_types_list": lambda w, ids, p, b: bamboohr_timeoff_types_list(w),
    "timeoff_policies_assign": lambda w, ids, p, b: bamboohr_timeoff_policies_assign(
        w, employee_id=ids[0], policies=b
    ),
    # Reports
    "reports_company": lambda w, ids, p, b: bamboohr_reports_company(w, report_id=ids[0], **p),
    "reports_custom": lambda w, ids, p, b: bamboohr_reports_custom(w, **{**p, **b}),
    # Goals
    "goals_status_count": lambda w, ids, p, b: bamboohr_goals_status_count(
        w, employee_id=ids[0], **p
    ),
    "goals_aggregate": lambda w, ids, p, b: bamboohr_goals_aggregate(w, employee_id=ids[0], **p),
    "goals_list": lambda w, ids, p, b: bamboohr_goals_list(w, employee_id=ids[0], **p),
    "goals_create": lambda w, ids, p, b: bamboohr_goals_create(w, employee_id=ids[0], **b),
    "goals_update": lambda w, ids, p, b: bamboohr_goals_update(
        w, employee_id=ids[0], goal_id=ids[1], **b
    ),
    "goals_delete": lambda w, ids, p, b: bamboohr_goals_delete(
        w, employee_id=ids[0], goal_id=ids[1]
    ),
    # Training
    "training_types_list": lambda w, ids, p, b: bamboohr_training_types_list(w),
    "training_types_create": lambda w, ids, p, b: bamboohr_training_types_create(w, **b),
    "training_types_update": lambda w, ids, p, b: bamboohr_training_types_update(
        w, training_type_id=ids[0], **b
    ),
    "training_types_delete": lambda w, ids, p, b: bamboohr_training_types_delete(
        w, training_type_id=ids[0]
    ),
    "training_categories_list": lambda w, ids, p, b: bamboohr_training_categories_list(w),
    "training_categories_create": lambda w, ids, p, b: bamboohr_training_categories_create(w, **b),
    "training_categories_update": lambda w, ids, p, b: bamboohr_training_categories_update(
        w, category_id=ids[0], **b
    ),
    "training_categories_delete": lambda w, ids, p, b: bamboohr_training_categories_delete(
        w, category_id=ids[0]
    ),
    "training_records_employee": lambda w, ids, p, b: bamboohr_training_records_employee(
        w, employee_id=ids[0], **p
    ),
    "training_records_create": lambda w, ids, p, b: bamboohr_training_records_create(
        w, employee_id=ids[0], **b
    ),
    "training_records_update": lambda w, ids, p, b: bamboohr_training_records_update(
        w, employee_id=ids[0], record_id=ids[1], **b
    ),
    "training_records_delete": lambda w, ids, p, b: bamboohr_training_records_delete(
        w, employee_id=ids[0], record_id=ids[1]
    ),
}


route_bamboohr = make_router(_ROUTES, _HANDLERS)
