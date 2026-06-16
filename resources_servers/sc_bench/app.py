# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.sc_bench.compare import compute_episode_reward_full
from resources_servers.sc_bench.supchain_tools import (
    TOOL_REGISTRY,
    configure_data_dir,
    get_data_dir,
)
from resources_servers.sc_bench.verify_utils import get_verifier_fields


class ScBenchResourcesServerConfig(BaseResourcesServerConfig):
    data_dir: str = ""


class QueryBuyerAndRelatedRequest(BaseModel):
    order_id: str


class FulfillmentIdRequest(BaseModel):
    fulfillment_id: str


class WarehouseStatusRequest(BaseModel):
    fulfillment_id: str
    warehouse_order_id: str


class ScBenchVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = Field(default=None)


class ScBenchVerifyResponse(BaseVerifyResponse):
    line_match_rate: float = 0.0
    num_gt_lines: int = 0


class ScBenchResourcesServer(SimpleResourcesServer):
    config: ScBenchResourcesServerConfig

    @staticmethod
    def _resolve_data_dir(data_dir: Path) -> Path:
        if data_dir.is_absolute():
            return data_dir
        return (PARENT_DIR / data_dir).resolve()

    def model_post_init(self, context: Any) -> None:
        data_dir = get_data_dir()
        if self.config.data_dir:
            data_dir = self._resolve_data_dir(Path(self.config.data_dir))
        if not (data_dir / "TradeOrders.csv").exists():
            raise FileNotFoundError(
                f"SC-bench CSV data not found at {data_dir}. "
                'Run: ng_prepare_benchmark "+config_paths=[benchmarks/sc_bench/config.yaml]"'
            )
        configure_data_dir(data_dir)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/query_buyer_and_related")(self.query_buyer_and_related)
        app.post("/get_fulfillment_status")(self.get_fulfillment_status)
        app.post("/get_cancel_scenes")(self.get_cancel_scenes)
        app.post("/get_cancel_error_code")(self.get_cancel_error_code)
        app.post("/get_error_reason")(self.get_error_reason)
        app.post("/check_fake_shipping")(self.check_fake_shipping)
        app.post("/get_warehouse_status")(self.get_warehouse_status)
        app.post("/get_warehouse_error_details")(self.get_warehouse_error_details)
        return app

    async def _invoke_tool(self, name: str, body: BaseModel) -> Dict[str, Any]:
        try:
            fn = TOOL_REGISTRY[name]
            return fn(**body.model_dump())
        except Exception as e:
            return {"error": f"Tool {name} failed: {e!s}"}

    async def query_buyer_and_related(self, body: QueryBuyerAndRelatedRequest) -> Dict[str, Any]:
        return await self._invoke_tool("query_buyer_and_related", body)

    async def get_fulfillment_status(self, body: FulfillmentIdRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_fulfillment_status", body)

    async def get_cancel_scenes(self, body: FulfillmentIdRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_cancel_scenes", body)

    async def get_cancel_error_code(self, body: FulfillmentIdRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_cancel_error_code", body)

    async def get_error_reason(self, body: FulfillmentIdRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_error_reason", body)

    async def check_fake_shipping(self, body: FulfillmentIdRequest) -> Dict[str, Any]:
        return await self._invoke_tool("check_fake_shipping", body)

    async def get_warehouse_status(self, body: WarehouseStatusRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_warehouse_status", body)

    async def get_warehouse_error_details(self, body: WarehouseStatusRequest) -> Dict[str, Any]:
        return await self._invoke_tool("get_warehouse_error_details", body)

    async def verify(self, body: ScBenchVerifyRequest) -> ScBenchVerifyResponse:
        gt_lines, expected_result = get_verifier_fields(body.verifier_metadata)
        if not gt_lines and not expected_result:
            return ScBenchVerifyResponse(**body.model_dump(), reward=0.0, line_match_rate=0.0, num_gt_lines=0)

        reward = compute_episode_reward_full(
            body.response.output,
            gt_lines,
            expected_result or None,
        )
        return ScBenchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            line_match_rate=float(reward),
            num_gt_lines=len(gt_lines),
        )


if __name__ == "__main__":
    ScBenchResourcesServer.run_webserver()
