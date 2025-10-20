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
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

import textworld
from textworld import EnvInfos
from textworld.core import Environment

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class TextworldResourcesServerConfig(BaseResourcesServerConfig):
    expose_admissible_commands: bool = False


class TextworldSeedSessionRequest(BaseSeedSessionRequest):
    game_file: str 


class TextworldSeedSessionResponse(BaseSeedSessionResponse):
    initial_observation: str
    objective: str
    admissible_commands: list[str] | None = None


class ExecuteCommandRequest(BaseModel):
    command: str


class ExecuteCommandResponse(BaseModel):
    observation: str
    score: int
    done: bool
    won: bool
    admissible_commands: list[str] | None = None


class TextworldVerifyRequest(BaseVerifyRequest):
    pass


class TextworldResourcesServer(SimpleResourcesServer):
    config: TextworldResourcesServerConfig
    session_id_to_env: Dict[str, Environment] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/execute_command")(self.execute_command)

        return app

    async def seed_session(
        self, request: Request, body: TextworldSeedSessionRequest
    ) -> TextworldSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        request_infos = EnvInfos(
            feedback=True,
            won=True,
            lost=True,
            score=True,
            max_score=True,
            objective=True,
            admissible_commands=self.config.expose_admissible_commands,
        )

        from pathlib import Path

        games_dir = Path(__file__).parent / "games"

        # Try direct path first
        game_path = games_dir / body.game_file

        # Hacky: if game not found, search in train/val/test subdirectories TODO: Fix paths
        if not game_path.exists():
            for split in ["train", "val", "test"]:
                for game_type in ["coin_collector", "treasure_hunter", "simple", "cooking", "custom"]:
                    potential_path = games_dir / split / game_type / body.game_file
                    if potential_path.exists():
                        game_path = potential_path
                        break
                if game_path.exists():
                    break

        if not game_path.exists():
            raise FileNotFoundError(f"Game file not found: {body.game_file} (searched in {games_dir})")

        env = textworld.start(str(game_path), request_infos=request_infos)

        state = env.reset()

        self.session_id_to_env[session_id] = env

        response = TextworldSeedSessionResponse(
            initial_observation=state.feedback if hasattr(state, "feedback") else str(state),
            objective=state["objective"] if "objective" in state else "",
        )

        if self.config.expose_admissible_commands and "admissible_commands" in state:
            response.admissible_commands = state["admissible_commands"]

        return response

    async def execute_command(
        self, request: Request, body: ExecuteCommandRequest
    ) -> ExecuteCommandResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        env = self.session_id_to_env[session_id]

        state, score, done = env.step(body.command)

        response = ExecuteCommandResponse(
            observation=state.feedback if hasattr(state, "feedback") else str(state),
            score=score,
            done=done,
            won=state.get("won", False),
        )

        if self.config.expose_admissible_commands and "admissible_commands" in state:
            response.admissible_commands = state["admissible_commands"]

        return response

    async def verify(self, request: Request, body: TextworldVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        reward = 0.0
        if session_id in self.session_id_to_env:
            env = self.session_id_to_env[session_id]
            if hasattr(env, "state") and env.state.get("won", False):
                reward = 1.0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    TextworldResourcesServer.run_webserver()
