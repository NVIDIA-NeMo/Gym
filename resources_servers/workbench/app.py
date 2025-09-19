# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.workbench.utils import is_correct
from resources_servers.workbench.workbench_tools.analytics import (
    AnalyticsTool,
)
from resources_servers.workbench.workbench_tools.calendar import (
    CalendarTool,
)
from resources_servers.workbench.workbench_tools.company_directory import (
    CompanyDirectoryTool,
)
from resources_servers.workbench.workbench_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
)
from resources_servers.workbench.workbench_tools.email import (
    EmailTool,
)
from resources_servers.workbench.workbench_tools.project_management import (
    ProjectManagementTool,
)


REASONING_TAG = os.getenv("REASONING_TAG", "think")


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    config: WorkbenchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)

        return app

    async def route_to_python_function(self, path: str, body: WorkbenchRequest) -> WorkbenchResponse:
        tool_name_to_class_to_function_mapping = {
            "company_directory_find_email_address": {
                "class": CompanyDirectoryTool,
                "function": "find_email_address",
            },
            "email_get_email_information_by_id": {
                "class": EmailTool,
                "function": "get_email_information_by_id",
            },
            "email_search_emails": {"class": EmailTool, "function": "search_emails"},
            "email_send_email": {"class": EmailTool, "function": "send_email"},
            "email_delete_email": {"class": EmailTool, "function": "delete_email"},
            "email_forward_email": {"class": EmailTool, "function": "forward_email"},
            "email_reply_email": {"class": EmailTool, "function": "reply_email"},
            "calendar_get_event_information_by_id": {
                "class": CalendarTool,
                "function": "get_event_information_by_id",
            },
            "calendar_search_events": {
                "class": CalendarTool,
                "function": "search_events",
            },
            "calendar_create_event": {
                "class": CalendarTool,
                "function": "create_event",
            },
            "calendar_delete_event": {
                "class": CalendarTool,
                "function": "delete_event",
            },
            "calendar_update_event": {
                "class": CalendarTool,
                "function": "update_event",
            },
            "analytics_engaged_users_count": {
                "class": AnalyticsTool,
                "function": "engaged_users_count",
            },
            "analytics_get_visitor_information_by_id": {
                "class": AnalyticsTool,
                "function": "get_visitor_information_by_id",
            },
            "analytics_create_plot": {
                "class": AnalyticsTool,
                "function": "create_plot",
            },
            "analytics_traffic_source_count": {
                "class": AnalyticsTool,
                "function": "traffic_source_count",
            },
            "analytics_total_visits_count": {
                "class": AnalyticsTool,
                "function": "total_visits_count",
            },
            "analytics_get_average_session_duration": {
                "class": AnalyticsTool,
                "function": "get_average_session_duration",
            },
            "project_management_get_task_information_by_id": {
                "class": ProjectManagementTool,
                "function": "get_task_information_by_id",
            },
            "project_management_search_tasks": {
                "class": ProjectManagementTool,
                "function": "search_tasks",
            },
            "project_management_create_task": {
                "class": ProjectManagementTool,
                "function": "create_task",
            },
            "project_management_delete_task": {
                "class": ProjectManagementTool,
                "function": "delete_task",
            },
            "project_management_update_task": {
                "class": ProjectManagementTool,
                "function": "update_task",
            },
            "customer_relationship_manager_search_customers": {
                "class": CustomerRelationshipManagerTool,
                "function": "search_customers",
            },
            "customer_relationship_manager_update_customer": {
                "class": CustomerRelationshipManagerTool,
                "function": "update_customer",
            },
            "customer_relationship_manager_add_customer": {
                "class": CustomerRelationshipManagerTool,
                "function": "add_customer",
            },
            "customer_relationship_manager_delete_customer": {
                "class": CustomerRelationshipManagerTool,
                "function": "delete_customer",
            },
        }

        class_function_mapping = tool_name_to_class_to_function_mapping.get(path)
        if not class_function_mapping:
            raise HTTPException(status_code=404, detail="Class not found")

        class_object = class_function_mapping["class"]()

        method_name = class_function_mapping["function"]  # string, e.g. "search_emails"

        fn = getattr(class_object, method_name, None)  # bound method on the instance
        if fn is None or not callable(fn):
            raise HTTPException(status_code=404, detail=f"Method {method_name} not found")

        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            result = fn(**args)  # sync tool method
            return WorkbenchResponse(output=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        ground_truth = body.ground_truth
        response = body.response.output
        total_score = 0.0

        # Convert list of ResponseFunctionToolCall objects into list of dictionaries
        predicted_function_calls = []
        for message in response:
            if message.type == "function_call":
                predicted_function_calls.append(message.model_dump())

        predicted_chat_content = []
        for message in response:
            if message.type == "output_text":
                predicted_chat_content.append(message.model_dump())

        # Use a single reward for correctness
        total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
        return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)


if __name__ == "__main__":
    server, app = WorkbenchResourcesServer.create_server()
    server.run_webserver()
elif __name__ == "app":  # for multiprocessing
    _, app = WorkbenchResourcesServer.create_server()
