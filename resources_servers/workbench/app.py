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
from typing import Dict, Any
from fastapi.responses import JSONResponse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

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

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workbench.utils import get_tools

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
    print('init tool env empty')
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> JSONResponse:
        app = super().setup_webserver()
        app.post("/{path}")(self.handle_tool_request)
        return app
    
    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
        ]
        print('init tool env')
        self.session_id_to_tool_env[session_id] = get_tools(toolkits)
        return BaseSeedSessionResponse()

    async def handle_tool_request(self, path: str, body: WorkbenchRequest, request: Request) -> JSONResponse:
        return await self.route_to_python_function(path, body, request)

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> JSONResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        # Check if session exists
        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(status_code=400, detail="Session not initialized. Please call seed_session first.")
        
        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        # Check if function exists in tool_env
        if path not in tool_env["functions"]:
            raise HTTPException(status_code=404, detail=f"Function {path} not found")
        
        try:
            function = tool_env["functions"][path]
            result = function(**args)
            print(result)
            # return WorkbenchResponse(output=result)
            
            return JSONResponse(content=result)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        ground_truth = body.ground_truth
        response = body.response.output

        from devtools import pprint
        pprint(response)
#         response =  [
#     {
#       "arguments": "{\"name\":\"carlos\"}",
#       "call_id": "call_BjosIz55T1NHLH0LZjhB9nBa",
#       "name": "company_directory_find_email_address",
#       "type": "function_call",
#       "id": "call_BjosIz55T1NHLH0LZjhB9nBa",
#       "status": "completed",
#       "output": None,
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": None,
#       "call_id": "call_BjosIz55T1NHLH0LZjhB9nBa",
#       "name": None,
#       "type": "function_call_output",
#       "id": None,
#       "status": None,
#       "output": "{\"output\":[\"carlos.rodriguez@atlas.com\"]}",
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": "{\"name\":\"yuki\"}",
#       "call_id": "call_Mgjf6NVNefSbSvccY5epZUZc",
#       "name": "company_directory_find_email_address",
#       "type": "function_call",
#       "id": "call_Mgjf6NVNefSbSvccY5epZUZc",
#       "status": "completed",
#       "output": None,
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": None,
#       "call_id": "call_Mgjf6NVNefSbSvccY5epZUZc",
#       "name": None,
#       "type": "function_call_output",
#       "id": None,
#       "status": None,
#       "output": "{\"output\":[\"yuki.tanaka@atlas.com\"]}",
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": "{\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\"}",
#       "call_id": "call_zRQquualhyNQnodv4zDDDvkq",
#       "name": "project_management_search_tasks",
#       "type": "function_call",
#       "id": "call_zRQquualhyNQnodv4zDDDvkq",
#       "status": "completed",
#       "output": None,
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": None,
#       "call_id": "call_zRQquualhyNQnodv4zDDDvkq",
#       "name": None,
#       "type": "function_call_output",
#       "id": None,
#       "status": None,
#       "output": "{\"output\":[{\"task_id\":\"00000039\",\"task_name\":\"Fix bug in data storage module\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-11-30\",\"board\":\"Back end\"},{\"task_id\":\"00000014\",\"task_name\":\"Implement login system API\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-01\",\"board\":\"Back end\"},{\"task_id\":\"00000077\",\"task_name\":\"Implement report generation API\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-01\",\"board\":\"Back end\"},{\"task_id\":\"00000089\",\"task_name\":\"Add authentication for cloud storage\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-02\",\"board\":\"Back end\"},{\"task_id\":\"00000081\",\"task_name\":\"Optimize database query for search functionality\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-03\",\"board\":\"Back end\"},{\"task_id\":\"00000083\",\"task_name\":\"Fix bug in user management module\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-04\",\"board\":\"Back end\"},{\"task_id\":\"00000032\",\"task_name\":\"Implement search functionality API\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-04\",\"board\":\"Back end\"},{\"task_id\":\"00000024\",\"task_name\":\"Update Flask to latest version\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-04\",\"board\":\"Back end\"},{\"task_id\":\"00000099\",\"task_name\":\"Fix bug in content delivery module\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-04\",\"board\":\"Back end\"},{\"task_id\":\"00000021\",\"task_name\":\"Update react to latest version\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-06\",\"board\":\"Back end\"},{\"task_id\":\"00000063\",\"task_name\":\"Optimize database query for report generation\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-08\",\"board\":\"Back end\"},{\"task_id\":\"00000082\",\"task_name\":\"Optimize database query for data export\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-08\",\"board\":\"Back end\"},{\"task_id\":\"00000023\",\"task_name\":\"Update Node.js to latest version\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-10\",\"board\":\"Back end\"},{\"task_id\":\"00000046\",\"task_name\":\"Fix bug in user authentication module\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-12\",\"board\":\"Back end\"},{\"task_id\":\"00000084\",\"task_name\":\"Add authentication for file upload\",\"assigned_to_email\":\"carlos.rodriguez@atlas.com\",\"list_name\":\"Backlog\",\"due_date\":\"2023-12-13\",\"board\":\"Back end\"}]}",
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": "{\"task_id\":\"00000039\",\"field\":\"assigned_to_email\",\"new_value\":\"yuki.tanaka@atlas.com\"}",
#       "call_id": "call_7m2TNJY0RVt9plpjcXQugak1",
#       "name": "project_management_update_task",
#       "type": "function_call",
#       "id": "call_7m2TNJY0RVt9plpjcXQugak1",
#       "status": "completed",
#       "output": None,
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": None,
#       "call_id": "call_7m2TNJY0RVt9plpjcXQugak1",
#       "name": None,
#       "type": "function_call_output",
#       "id": None,
#       "status": None,
#       "output": "{\"output\":\"Task updated successfully.\"}",
#       "content": None,
#       "role": None
#     },
#     {
#       "arguments": None,
#       "call_id": None,
#       "name": None,
#       "type": "message",
#       "id": "msg_0b4702f2181b4661aacd188d1581d36a",
#       "status": "completed",
#       "output": None,
#       "content": [
#         {
#           "annotations": [],
#           "text": "Here are Carlos's overdue tasks that he hasn't started (tasks in \"Backlog\" with due dates before today):\n\n- Fix bug in data storage module (Due 2023-11-30)\n\nI am reassigning these overdue tasks from Carlos to Yuki now.\n\nWould you like to reassign any more tasks or take further action?",
#           "type": "output_text",
#           "logprobs": None
#         }
#       ],
#       "role": "assistant"
#     }
#   ]
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
#         ground_truth = [
#   {
#     "name": "project_management_update_task",
#     "arguments": "{\"task_id\": \"00000039\", \"field\": \"assigned_to_email\", \"new_value\": \"yuki.tanaka@atlas.com\"}"
#   }
# ]
        total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
        print("total_score: ", total_score, '\n\n')
        return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
