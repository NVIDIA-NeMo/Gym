# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from resources_servers.sc_bench.supchain_tools import (
    _map_status,
    _parse_buyer,
    check_fake_shipping,
    clear_csv_cache,
    configure_data_dir,
    get_cancel_error_code,
    get_cancel_scenes,
    get_data_dir,
    get_error_reason,
    get_fulfillment_status,
    get_warehouse_error_details,
    get_warehouse_status,
    query_buyer_and_related,
    to_nemo_gym_tools,
)


class TestSupchainTools:
    def test_to_nemo_gym_tools(self) -> None:
        tools = to_nemo_gym_tools()
        assert len(tools) == 8
        assert tools[0]["type"] == "function"
        assert "name" in tools[0]

    def test_csv_cache_reuses_dataframes(self, configure_csv_dir) -> None:
        data_dir = get_data_dir()
        clear_csv_cache()
        configure_data_dir(data_dir)
        first = query_buyer_and_related("T1001")
        second = query_buyer_and_related("T1001")
        assert first == second

    def test_unknown_trade_order(self, configure_csv_dir) -> None:
        result = query_buyer_and_related("T99999")
        assert result["buyer_id"] is None
        assert result["related_item"] == []

    def test_fulfillment_and_warehouse_status(self, configure_csv_dir) -> None:
        related = query_buyer_and_related("T1001")
        fid = related["related_item"][0]["fulfillment_id"]
        wid = related["related_item"][0]["warehouse_order_id"]
        status = get_fulfillment_status(fid)
        wh_status = get_warehouse_status(fid, wid)
        assert "status" in status
        assert "status" in wh_status

    def test_configure_data_dir_clears_cache(self, configure_csv_dir) -> None:
        data_dir = get_data_dir()
        query_buyer_and_related("T1001")
        clear_csv_cache()
        configure_data_dir(data_dir)
        assert query_buyer_and_related("T1001")["buyer_id"] is not None

    def test_map_status_variants(self) -> None:
        assert _map_status("RECEIVING") == "packing_in_progress"
        assert _map_status("PACKED") == "packing_done"
        assert _map_status("SHIPPED") == "dispatched"
        assert _map_status("IN_TRANSIT") == "in_transit"
        assert _map_status("DELIVERED") == "delivered"
        assert _map_status("ERROR") == "error"
        assert _map_status("UNKNOWN") == "packing_in_progress"

    def test_parse_buyer_variants(self) -> None:
        import pandas as pd

        assert _parse_buyer(pd.NA) is None
        assert _parse_buyer({"id": 1}) == {"id": 1}
        assert _parse_buyer('{"id": 90000}') == {"id": 90000}
        assert _parse_buyer("plain-text") == "plain-text"

    def test_fulfillment_status_paths(self, configure_csv_dir) -> None:
        assert get_fulfillment_status("FO2001")["status"] == "cancelled"
        assert get_fulfillment_status("FO2018")["status"] == "error"
        assert get_fulfillment_status("FO2009")["status"] == "error"
        assert get_fulfillment_status("FO2003")["status"] == "in_transit"
        assert get_fulfillment_status("FO2008")["status"] == "dispatched"
        assert get_fulfillment_status("FO2022")["status"] == "delivered"
        assert get_fulfillment_status("FO2014")["status"] == "packing_done"
        assert get_fulfillment_status("FO2007")["status"] == "packing_in_progress"

    def test_cancellation_and_error_tools(self, configure_csv_dir) -> None:
        assert get_cancel_scenes("FO2001")["cancelType"] == "BUYER"
        assert get_cancel_scenes("FO9999") == {"cancelType": None}
        cancel_err = get_cancel_error_code("FO2001")
        assert "size received" in str(cancel_err["cancelErrorMsg"])
        assert get_cancel_error_code("FO9999") == {"cancelErrorCode": None, "cancelErrorMsg": None}
        err = get_error_reason("FO2018")
        assert err["code"] == "FAKE_SHIP"
        assert get_error_reason("FO9999") == {"code": None, "text": None}

    def test_fake_shipping_and_warehouse_errors(self, configure_csv_dir) -> None:
        assert check_fake_shipping("FO2018") == {"exceptionFlag": True}
        assert check_fake_shipping("FO2001") == {"exceptionFlag": False}
        details = get_warehouse_error_details("FO2009", "WO3016")
        assert details["code"] == "PACK_MISSING_LABEL"
        assert get_warehouse_error_details("FO9999", "WO9999") == {"code": None, "text": None}
        assert get_warehouse_status("FO9999", "WO9999") == {"status": None, "error": None}
