# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.sc_bench.app import (
    QueryBuyerAndRelatedRequest,
    ScBenchResourcesServer,
    ScBenchResourcesServerConfig,
    ScBenchVerifyRequest,
    WarehouseStatusRequest,
)


@pytest.fixture
def server(configure_csv_dir):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "csv"
    if not (data_dir / "TradeOrders.csv").exists():
        data_dir = Path(__file__).resolve().parents[3] / "SC-bench" / "data"
    config = ScBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sc_bench",
        data_dir=str(data_dir),
    )
    return ScBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestDataDir:
    def test_resolve_relative_data_dir(self) -> None:
        resolved = ScBenchResourcesServer._resolve_data_dir(Path("resources_servers/sc_bench/data/csv"))
        assert resolved == (PARENT_DIR / "resources_servers/sc_bench/data/csv").resolve()
        assert (resolved / "TradeOrders.csv").exists()

    def test_missing_csv_raises_on_init(self) -> None:
        config = ScBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="sc_bench",
            data_dir="/nonexistent/sc_bench/csv",
        )
        with pytest.raises(FileNotFoundError, match="SC-bench CSV data not found"):
            ScBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestTools:
    @pytest.mark.asyncio
    async def test_query_buyer_and_related(self, server) -> None:
        result = await server.query_buyer_and_related(QueryBuyerAndRelatedRequest(order_id="T1001"))
        assert result["buyer_id"] is not None
        assert result["related_item"]

    @pytest.mark.asyncio
    async def test_all_tool_endpoints(self, server) -> None:
        from resources_servers.sc_bench.app import FulfillmentIdRequest

        related = await server.query_buyer_and_related(QueryBuyerAndRelatedRequest(order_id="T1001"))
        fid = related["related_item"][0]["fulfillment_id"]
        wid = related["related_item"][0]["warehouse_order_id"]
        assert "status" in await server.get_fulfillment_status(FulfillmentIdRequest(fulfillment_id=fid))
        assert await server.get_cancel_scenes(FulfillmentIdRequest(fulfillment_id=fid)) is not None
        assert await server.get_cancel_error_code(FulfillmentIdRequest(fulfillment_id=fid)) is not None
        assert await server.get_error_reason(FulfillmentIdRequest(fulfillment_id=fid)) is not None
        assert await server.check_fake_shipping(FulfillmentIdRequest(fulfillment_id=fid)) is not None
        assert "status" in await server.get_warehouse_status(
            WarehouseStatusRequest(fulfillment_id=fid, warehouse_order_id=wid)
        )
        assert (
            await server.get_warehouse_error_details(
                WarehouseStatusRequest(fulfillment_id=fid, warehouse_order_id=wid)
            )
            is not None
        )

    @pytest.mark.asyncio
    async def test_tool_error_returns_json(self, server) -> None:
        def boom(**_kwargs):
            raise RuntimeError("boom")

        with patch("resources_servers.sc_bench.app.TOOL_REGISTRY", {"get_warehouse_status": boom}):
            result = await server.get_warehouse_status(
                WarehouseStatusRequest(fulfillment_id="FO2001", warehouse_order_id="WO3001")
            )
        assert "error" in result


class TestVerify:
    @staticmethod
    def _tool_trace_output():
        tool_trace = [
            {
                "step": 1,
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1001"},
                "output": {
                    "buyer_id": {"id": 90000},
                    "related_item": [{"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"}],
                },
            },
            {
                "step": 2,
                "name": "get_cancel_error_code",
                "arguments": {"fulfillment_id": "FO2001"},
                "output": {
                    "cancelErrorCode": "SIZE_MISMATCH",
                    "cancelErrorMsg": "Because the size received was incorrect, the buyer had no choice but to cancel and reorder the proper one.",
                },
            },
            {
                "step": 3,
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"},
                "output": {"status": "packing_in_progress", "error": None},
            },
        ]
        output = []
        for step in tool_trace:
            call_id = f"call_{step['step']}"
            output.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": step["name"],
                    "arguments": json.dumps(step["arguments"]),
                }
            )
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(step["output"]),
                }
            )
        return output

    @pytest.mark.asyncio
    async def test_verify_matching_tool_trace(self, server) -> None:
        request = ScBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "T1001"}]},
            response=NeMoGymResponse(
                id="",
                object="response",
                created_at=0.0,
                model="test",
                output=self._tool_trace_output(),
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
            verifier_metadata={
                "trade_order_id": "T1001",
                "gt_lines": [
                    {
                        "trade_order_id": "T1001",
                        "fulfillment_id": "FO2001",
                        "cancel_scene": None,
                        "buyer_cancel_reason": "Because the size received was incorrect, the buyer had no choice but to cancel and reorder the proper one.",
                        "warehouse_order_id": "WO3001",
                        "warehouse_order_status": "packing_in_progress",
                    }
                ],
                "expected_result": {},
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_wrong_tool_trace(self, server) -> None:
        request = ScBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "T1001"}]},
            response=NeMoGymResponse(
                id="",
                object="response",
                created_at=0.0,
                model="test",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "query_buyer_and_related",
                        "arguments": json.dumps({"order_id": "T1001"}),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": json.dumps({"buyer_id": {"id": 1}, "related_item": []}),
                    },
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
            verifier_metadata={
                "gt_lines": [
                    {
                        "trade_order_id": "T1001",
                        "fulfillment_id": "FO2001",
                        "warehouse_order_id": "WO3001",
                        "warehouse_order_status": "delivered",
                    }
                ],
                "expected_result": {},
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_empty_output(self, server) -> None:
        request = ScBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "T1001"}]},
            response=NeMoGymResponse(
                id="",
                object="response",
                created_at=0.0,
                model="test",
                output=[],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
            verifier_metadata={"gt_lines": [{"trade_order_id": "T1001"}], "expected_result": {}},
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    def test_setup_webserver_registers_routes(self, server) -> None:
        app = server.setup_webserver()
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/query_buyer_and_related" in paths
        assert "/get_warehouse_error_details" in paths

    @pytest.mark.asyncio
    async def test_verify_missing_metadata(self, server) -> None:
        request = ScBenchVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "T1001"}]},
            response=NeMoGymResponse(
                id="",
                object="response",
                created_at=0.0,
                model="test",
                output=[],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
        )
        result = await server.verify(request)
        assert result.reward == 0.0
