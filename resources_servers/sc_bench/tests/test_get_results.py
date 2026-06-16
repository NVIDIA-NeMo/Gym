# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from resources_servers.sc_bench.get_results import (
    _group_warehouse_ids,
    extract_trade_order_id,
    get_results,
    get_results_standard,
    process_fulfillment,
    process_fulfillment_standard,
)


class TestExtractTradeOrderId:
    def test_empty_question(self) -> None:
        assert extract_trade_order_id("") is None
        assert extract_trade_order_id("no order here") is None

    def test_extracts_tid(self) -> None:
        assert extract_trade_order_id("Check trade order t1001 please") == "T1001"


class TestGroupWarehouseIds:
    def test_groups_and_keeps_fulfillment_without_warehouse(self) -> None:
        grouped = _group_warehouse_ids(
            [
                {"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"},
                {"fulfillment_id": "FO2002", "warehouse_order_id": None},
            ]
        )
        assert grouped["FO2001"] == ["WO3001"]
        assert grouped["FO2002"] == []


class TestProcessFulfillment:
    def test_canceled_fulfillment(self, configure_csv_dir) -> None:
        result = process_fulfillment("FO2001", ["WO3001"], trade_order_id="T1001")
        assert result["biz_status"] == "canceled"
        assert result["cancel_type"] == "BUYER"
        assert "size received" in str(result["reason_text"])
        assert result["warehouse_orders"][0]["warehouse_order_id"] == "WO3001"

    def test_error_fulfillment(self, configure_csv_dir) -> None:
        result = process_fulfillment("FO2018", ["WO3034"], trade_order_id="T1006")
        assert result["biz_status"] == "error"
        assert result["code"] == "FAKE_SHIP"

    def test_unknown_cancel_type_checks_fake_shipping(self, configure_csv_dir) -> None:
        with patch(
            "resources_servers.sc_bench.get_results.get_cancel_scenes",
            return_value={"cancelType": "other"},
        ):
            result = process_fulfillment("FO2001", ["WO3001"], trade_order_id="T1001")
        assert result["biz_status"] == "canceled"


class TestProcessFulfillmentStandard:
    def test_builds_standard_records(self, configure_csv_dir) -> None:
        fulfillment, warehouse_entries = process_fulfillment_standard("FO2001", "T1001", ["WO3001"])
        assert fulfillment["status"] == "canceled"
        assert fulfillment["cancel_type"] == "BUYER"
        assert warehouse_entries[0]["warehouse_order_id"] == "WO3001"

    def test_error_branch(self, configure_csv_dir) -> None:
        fulfillment, _ = process_fulfillment_standard("FO2018", "T1006", ["WO3034"])
        assert fulfillment["status"] == "error"
        assert fulfillment["errorCode"] == "FAKE_SHIP"

    def test_canceled_without_cancel_type(self, configure_csv_dir) -> None:
        with patch(
            "resources_servers.sc_bench.get_results.get_cancel_scenes",
            return_value={"cancelType": None},
        ):
            fulfillment, _ = process_fulfillment_standard("FO2001", "T1001", ["WO3001"])
        assert fulfillment["status"] == "canceled"
        assert fulfillment["cancel_type"] is None


class TestGetResults:
    def test_t1001_canceled_order(self, configure_csv_dir) -> None:
        question = "For trade order T1001, what was the cancellation reason for FO2001?"
        result = get_results(question)
        assert result["trade_order_id"] == "T1001"
        assert result["buyer_id"] is not None
        assert any(f["fulfillment_order_id"] == "FO2001" for f in result["fulfillments"])

    def test_t1003_in_progress_order(self, configure_csv_dir) -> None:
        question = "What is the status of warehouse order WO3013 for trade order T1003?"
        result = get_results(question)
        assert result["trade_order_id"] == "T1003"
        fo2007 = next(f for f in result["fulfillments"] if f["fulfillment_order_id"] == "FO2007")
        assert fo2007["biz_status"] == "packing_in_progress"

    def test_missing_trade_order_id(self, configure_csv_dir) -> None:
        result = get_results("No trade order id in this question.")
        assert result["trade_order_id"] is None
        assert result["error"] == "trade_order_id_not_found"


class TestGetResultsStandard:
    def test_t1001_standard_shape(self, configure_csv_dir) -> None:
        result = get_results_standard("Summarize trade order T1001.")
        assert result["trade_orders"][0]["trade_order_id"] == "T1001"
        assert result["fulfillment_orders"]
        assert result["warehouse_orders"]

    def test_missing_trade_order_id(self, configure_csv_dir) -> None:
        result = get_results_standard("No id here.")
        assert result["trade_orders"][0]["trade_order_id"] is None
        assert result["fulfillment_orders"] == []
