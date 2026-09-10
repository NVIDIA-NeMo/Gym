# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""bamboohr tools for AutomationBench."""

from automationbench.tools.zapier.bamboohr.actions import (
    bamboohr_add_timesheet_clock_entries,
    bamboohr_department,
    bamboohr_employee_file_category,
    bamboohr_employeeCreate,
    bamboohr_employeeList,
    bamboohr_employeeSearch,
    bamboohr_get_summary_of_whos_out,
    bamboohr_list_fields,
    bamboohr_list_monitor_fields,
    bamboohr_new_employee,
    bamboohr_report_changed,
    bamboohr_report_field,
    bamboohr_respond_to_time_off_request,
    bamboohr_terminated_employee,
    bamboohr_time_off,
    bamboohr_time_off_request,
    bamboohr_update_employee,
    bamboohr_updated_employee,
    bamboohr_updated_employee_poll,
    bamboohr_upload_employee_file,
)

__all__ = [
    "bamboohr_department",
    "bamboohr_employeeList",
    "bamboohr_employee_file_category",
    "bamboohr_list_fields",
    "bamboohr_list_monitor_fields",
    "bamboohr_new_employee",
    "bamboohr_report_changed",
    "bamboohr_report_field",
    "bamboohr_terminated_employee",
    "bamboohr_time_off",
    "bamboohr_time_off_request",
    "bamboohr_updated_employee",
    "bamboohr_updated_employee_poll",
    "bamboohr_employeeSearch",
    "bamboohr_add_timesheet_clock_entries",
    "bamboohr_employeeCreate",
    "bamboohr_get_summary_of_whos_out",
    "bamboohr_respond_to_time_off_request",
    "bamboohr_update_employee",
    "bamboohr_upload_employee_file",
]
