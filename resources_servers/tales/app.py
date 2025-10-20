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
import importlib
from typing import Dict

import gymnasium as gym
from fastapi import FastAPI, HTTPException, Request
from gymnasium import Env
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class TALESResourcesServerConfig(BaseResourcesServerConfig):
    expose_admissible_commands: bool = False
    framework: str = "alfworld"
    task_no: int = 0
    seed: int = 0
    split: str = "train"  # "train" or "test"
    max_episode_steps: int = 25  # If not provided, use default from config


class TALESVerifyRequest(BaseVerifyRequest):
    reward: float


class TALESSeedSessionResponse(BaseSeedSessionResponse):
    observation: str
    score: float
    done: bool
    info: dict
    session_id: str
    available_tasks: int
    admissible_commands: list[str] | None = None


class ExecuteCommandResponse(BaseModel):
    observation: str
    score: float
    done: bool
    info: dict
    admissible_commands: list[str] | None = None


class ExecuteCommandRequest(BaseModel):
    command: str
    session_id: str


class ExecuteResetResponse(BaseModel):
    observation: str
    infos: dict


class ExecuteResetRequest(BaseModel):
    pass


class TALESVerifyRequest(BaseVerifyRequest):
    pass


class TALESSeedSessionRequest(BaseSeedSessionRequest):
    framework: str | None = None
    task_no: int | None = None
    split: str | None = None  # "train" or "test"
    max_episode_steps: int | None = None  # If not provided, use default from config
    seed: int | None = None  # If not provided, use default from config


class TALESRequest(BaseModel):
    session_id: str | None = None
    command: str | None = None


class TALESResponse(BaseModel):
    observation: str
    score: float
    done: bool
    info: dict
    admissible_commands: list[str] | None = None


class TALESResourcesServer(SimpleResourcesServer):
    config: TALESResourcesServerConfig
    session_id_to_env: Dict[str, Env] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def route_to_python_function(self, path: str, body: TALESRequest, request: Request) -> TALESResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Check if session exists
        if session_id not in self.session_id_to_env and body.session_id not in self.session_id_to_env:
            raise HTTPException(
                status_code=400,
                detail=f"Session id: {session_id} not initialized. Please call seed_session first.",
            )
        if session_id != body.session_id:
            session_id = body.session_id

        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            if "command" in args.keys():
                return await self.execute_command(request, ExecuteCommandRequest(**args))
            else:
                return await self.reset(request)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error trying to execute command {args}, request body: {request}, Exception \n{e}\n",
            )

    async def seed_session(self, request: Request, body: TALESSeedSessionRequest) -> TALESSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Lazy importing to avoid slowdown from loading all of the environments if not used
        # Check if framework and other start info was sent in the request body, otherwise use the defaults from config
        if body.framework is not None:
            framework_path = f"tales.{body.framework}"
        else:
            framework_path = f"tales.{self.config.framework}"

        if body.task_no is not None:
            task_no = body.task_no
        else:
            task_no = self.config.task_no

        if body.split is not None:
            split = body.split
        else:
            split = self.config.split

        if body.seed is not None:
            self.config.seed = body.seed

        framework = importlib.import_module(framework_path)

        if split == "train":
            envs = framework.train_environments
        else:
            envs = framework.environments

        # Make sure the task number is within the range of available tasks. If not, return an error:
        if task_no < 0 or task_no >= len(envs):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task number {task_no}. Please choose a task number between 0 and {len(envs) - 1}.",
            )
        task = envs[task_no]

        env_key = f"{task[0]}-{task[1]}"
        env = gym.make(id=f"tales/{env_key}", disable_env_checker=True, admissible_commands=True)

        obs, info = env.reset()

        self.session_id_to_env[session_id] = env

        response = TALESSeedSessionResponse(
            observation=obs, done=False, score=0, info=info, session_id=session_id, available_tasks=len(envs)
        )

        if self.config.expose_admissible_commands and "admissible_commands" in info.keys():
            response.admissible_commands = info["admissible_commands"]

        return response

    async def reset(self, request: Request) -> ExecuteResetResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )
        env = self.session_id_to_env[session_id]
        obs, info = env.reset(self.config.seed)

        response = ExecuteResetResponse(
            observation=obs,
            score=0,
            done=False,
            info=info,
            session_id_key=SESSION_ID_KEY,
        )

        if self.config.expose_admissible_commands and "admissible_commands" in info:
            response.admissible_commands = info["admissible_commands"]

        return response

    async def execute_command(self, request: Request, body: ExecuteCommandRequest) -> ExecuteCommandResponse:
        session_id = request.session[SESSION_ID_KEY]
        session_id = body.session_id if body.session_id is not None else session_id
        if session_id not in self.session_id_to_env and body.session_id not in self.session_id_to_env:
            raise HTTPException(
                status_code=400,
                detail=f"Session id {session_id} and {body.session_id} not initialized. Please call seed_session first.",
            )

        env = self.session_id_to_env[session_id]

        obs, score, done, info = env.step(body.command)
        # print(f"Command: {body.command}\nObservation: {obs}\nScore: {score}\nDone: {done}\nInfo: {info}\n")

        response = ExecuteCommandResponse(
            observation=obs,
            score=score,
            done=done,
            info=info,
        )

        if self.config.expose_admissible_commands and "admissible_commands" in info:
            response.admissible_commands = info["admissible_commands"]

        return response

    async def verify(self, request: Request, body: TALESVerifyRequest) -> BaseVerifyResponse:
        reward = body.reward

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    TALESResourcesServer.run_webserver()
