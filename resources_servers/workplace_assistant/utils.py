# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from typing import Dict, List

from resources_servers.workplace_assistant.workplace_assistant_tools.analytics import (
    AnalyticsTool,
    analytics_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.calendar import (
    CalendarTool,
    calendar_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.company_directory import (
    CompanyDirectoryTool,
    company_directory_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
    customer_relationship_manager_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.email import (
    EmailTool,
    email_tool_schemas,
)
from resources_servers.workplace_assistant.workplace_assistant_tools.project_management import (
    ProjectManagementTool,
    project_management_tool_schemas,
)


def get_tools(toolkits):
    tool_env = {
        "containers": {},
        "functions": {},
        "schemas": [],
    }
    company_directory = CompanyDirectoryTool()
    tool_env["containers"]["company_directory"] = company_directory
    tool_env["functions"]["company_directory_find_email_address"] = company_directory.find_email_address
    tool_env["schemas"].extend(company_directory_tool_schemas)
    if "email" in toolkits:
        email = EmailTool()
        tool_env["containers"]["email"] = email
        tool_env["functions"]["email_get_email_information_by_id"] = email.get_email_information_by_id
        tool_env["functions"]["email_search_emails"] = email.search_emails
        tool_env["functions"]["email_send_email"] = email.send_email
        tool_env["functions"]["email_delete_email"] = email.delete_email
        tool_env["functions"]["email_forward_email"] = email.forward_email
        tool_env["functions"]["email_reply_email"] = email.reply_email
        tool_env["schemas"].extend(email_tool_schemas)
    if "calendar" in toolkits:
        calendar = CalendarTool()
        tool_env["containers"]["calendar"] = calendar
        tool_env["functions"]["calendar_get_event_information_by_id"] = calendar.get_event_information_by_id
        tool_env["functions"]["calendar_search_events"] = calendar.search_events
        tool_env["functions"]["calendar_create_event"] = calendar.create_event
        tool_env["functions"]["calendar_delete_event"] = calendar.delete_event
        tool_env["functions"]["calendar_update_event"] = calendar.update_event
        tool_env["schemas"].extend(calendar_tool_schemas)
    if "analytics" in toolkits:
        analytics = AnalyticsTool()
        tool_env["containers"]["analytics"] = analytics
        tool_env["functions"]["analytics_engaged_users_count"] = analytics.engaged_users_count
        tool_env["functions"]["analytics_get_visitor_information_by_id"] = analytics.get_visitor_information_by_id
        tool_env["functions"]["analytics_create_plot"] = analytics.create_plot
        tool_env["functions"]["analytics_traffic_source_count"] = analytics.traffic_source_count
        tool_env["functions"]["analytics_total_visits_count"] = analytics.total_visits_count
        tool_env["functions"]["analytics_get_average_session_duration"] = analytics.get_average_session_duration
        tool_env["schemas"].extend(analytics_tool_schemas)
    if "project_management" in toolkits:
        project_management = ProjectManagementTool()
        tool_env["containers"]["project_management"] = project_management
        tool_env["functions"]["project_management_get_task_information_by_id"] = (
            project_management.get_task_information_by_id
        )
        tool_env["functions"]["project_management_search_tasks"] = project_management.search_tasks
        tool_env["functions"]["project_management_create_task"] = project_management.create_task
        tool_env["functions"]["project_management_delete_task"] = project_management.delete_task
        tool_env["functions"]["project_management_update_task"] = project_management.update_task
        tool_env["schemas"].extend(project_management_tool_schemas)
    if "customer_relationship_manager" in toolkits:
        customer_relationship_manager = CustomerRelationshipManagerTool()
        tool_env["containers"]["customer_relationship_manager"] = customer_relationship_manager
        tool_env["functions"]["customer_relationship_manager_search_customers"] = (
            customer_relationship_manager.search_customers
        )
        tool_env["functions"]["customer_relationship_manager_update_customer"] = (
            customer_relationship_manager.update_customer
        )
        tool_env["functions"]["customer_relationship_manager_add_customer"] = (
            customer_relationship_manager.add_customer
        )
        tool_env["functions"]["customer_relationship_manager_delete_customer"] = (
            customer_relationship_manager.delete_customer
        )
        tool_env["schemas"].extend(customer_relationship_manager_tool_schemas)
    return tool_env


def execute_actions_and_reset_state(actions: List[Dict[str, str]]):
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]
    tool_env = get_tools(toolkits)

    # Execute the actions
    for action in actions:
        try:
            tool_env["functions"][action["name"]](**json.loads(action["arguments"]))
        except Exception as e:
            print("Error executing tool: ", e)
            continue
    return tool_env


def is_correct(predicted_actions: Dict[str, str], ground_truth_actions: Dict[str, str], error: str) -> bool:
    """
    Checks if the prediction is correct by comparing the state change after executing the actions.

    Parameters
    ----------
    predicted_actions : list
        List of predicted actions as strings.
    ground_truth_actions : list
        List of ground truth actions as strings.
    error : str
        Error message from the prediction.

    Returns
    -------
    bool
        True if the predicted actions result in the same state change as the ground truth actions.

    """
    if error:
        return False
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df):
        # For some fields the case matters, so we don't convert them to lowercase
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    # We allow for case-insensitive comparison of strings for most fields
    predicted_calendar_state = convert_strs_to_lowercase(predict_env["containers"]["calendar"]._calendar_events)
    predicted_email_state = convert_strs_to_lowercase(predict_env["containers"]["email"]._emails)
    predicted_analytics_state = convert_strs_to_lowercase(predict_env["containers"]["analytics"]._plots_data)
    predicted_project_management_state = convert_strs_to_lowercase(
        predict_env["containers"]["project_management"]._project_tasks
    )
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predict_env["containers"]["customer_relationship_manager"]._crm_data
    )

    ground_truth_calendar_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["calendar"]._calendar_events
    )
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_env["containers"]["email"]._emails)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_env["containers"]["analytics"]._plots_data)
    ground_truth_project_management_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["project_management"]._project_tasks
    )
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["customer_relationship_manager"]._crm_data
    )

    return (
        predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )
