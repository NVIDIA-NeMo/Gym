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

"""
Custom Tools Resources Server with Dynamic Verifier Delegation.

This resources server provides custom tool implementations while delegating
verification to other existing resources servers based on per-sample configuration.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef


# ============================================================
# Configuration
# ============================================================


class MyCustomToolsConfig(BaseResourcesServerConfig):
    """Config for the custom tools resources server."""

    # Map of verifier names to server references
    available_verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)

    # Default verifier if not specified in sample
    default_verifier: str = "xlam_fc"


# ============================================================
# Tool Request/Response Models
# ============================================================


class SearchDatabaseRequest(BaseModel):
    query: str
    database: str = "default"


class SearchDatabaseResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int


class ExecuteCalculationRequest(BaseModel):
    expression: str
    precision: int = 4


class ExecuteCalculationResponse(BaseModel):
    result: float
    formatted: str


class FetchDataRequest(BaseModel):
    source_id: str
    fields: List[str] = Field(default_factory=list)


class FetchDataResponse(BaseModel):
    data: Dict[str, Any]
    source: str


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class MyCustomToolsRunRequest(BaseRunRequest):
    """
    Run request that allows extra fields from the sample.
    The verifier_type field determines which verifier to use.
    """

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection
    verifier_type: Optional[str] = None

    # Fields that different verifiers might need
    expected_answers: Optional[List[Dict[str, Any]]] = None  # for xlam_fc
    expected_answer: Optional[str] = None  # for mcqa / equivalence_llm_judge
    options: Optional[List[Dict[str, str]]] = None  # for mcqa


class MyCustomToolsVerifyRequest(MyCustomToolsRunRequest, BaseVerifyRequest):
    pass


class MyCustomToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_used: str
    delegated_response: Optional[Dict[str, Any]] = None


# ============================================================
# Resources Server Implementation
# ============================================================


class MyCustomToolsResourcesServer(SimpleResourcesServer):
    config: MyCustomToolsConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register your custom tool endpoints
        app.post("/search_database")(self.search_database)
        app.post("/execute_calculation")(self.execute_calculation)
        app.post("/fetch_data")(self.fetch_data)

        return app

    # --------------------------------------------------------
    # Custom Tool Implementations
    # --------------------------------------------------------

    async def search_database(self, body: SearchDatabaseRequest) -> SearchDatabaseResponse:
        """Search a database with the given query."""
        # Mock implementation - replace with your actual logic
        results = [{"id": 1, "name": f"Result for: {body.query}", "database": body.database}]
        return SearchDatabaseResponse(results=results, total_count=len(results))

    async def execute_calculation(self, body: ExecuteCalculationRequest) -> ExecuteCalculationResponse:
        """Execute a mathematical calculation."""
        # Safe evaluation with limited builtins
        safe_builtins = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        try:
            result = eval(body.expression, {"__builtins__": safe_builtins}, {})
            result = round(float(result), body.precision)
        except Exception:
            result = 0.0
        return ExecuteCalculationResponse(result=result, formatted=f"{result:.{body.precision}f}")

    async def fetch_data(self, body: FetchDataRequest) -> FetchDataResponse:
        """Fetch data from a specified source."""
        # Mock implementation - replace with your actual logic
        data = {"source_id": body.source_id, "fields": body.fields, "fetched": True}
        return FetchDataResponse(data=data, source=body.source_id)

    # --------------------------------------------------------
    # Dynamic Verification
    # --------------------------------------------------------

    async def verify(self, body: MyCustomToolsVerifyRequest) -> MyCustomToolsVerifyResponse:
        """
        Verify the model's response using the appropriate verifier.
        The verifier is selected based on the sample's verifier_type field.
        """
        # Determine which verifier to use
        verifier_type = body.verifier_type or self.config.default_verifier

        # Check if the verifier is available
        if verifier_type not in self.config.available_verifiers:
            return MyCustomToolsVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verifier_used=f"unknown:{verifier_type}",
                delegated_response={
                    "error": f"Unknown verifier: {verifier_type}. "
                    f"Available: {list(self.config.available_verifiers.keys())}"
                },
            )

        verifier_ref = self.config.available_verifiers[verifier_type]

        # Delegate to the appropriate verifier
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),  # Pass through ALL fields
        )

        result = await response.json()
        reward = result.get("reward", 0.0)

        return MyCustomToolsVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verifier_used=verifier_type,
            delegated_response=result,
        )


if __name__ == "__main__":
    MyCustomToolsResourcesServer.run_webserver()

