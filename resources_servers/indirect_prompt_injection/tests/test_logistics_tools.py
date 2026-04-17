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
import copy

from resources_servers.indirect_prompt_injection.logistics_tools import (
    TOOL_HANDLERS,
    add_warehouse_inventory,
    adjust_warehouse_quantity,
    get_carrier_messages,
    get_customs_declaration_by_shipment,
    get_delivery_schedule,
    get_shipment_by_id,
    get_warehouse_inventory_item,
    list_low_stock_items,
    search_shipments_by_status,
    send_carrier_message,
    submit_customs_declaration,
    update_shipment_status,
)


SAMPLE_ENV = {
    "shipments": {
        "SH-1001": {
            "shipment_id": "SH-1001",
            "tracking_number": "TRK-99887",
            "origin": "Los Angeles, CA",
            "destination": "New York, NY",
            "status": "in_transit",
            "eta": "2025-02-01",
            "notes": "Fragile items.",
        },
    },
    "warehouse_inventories": {
        "INV-2001": {
            "inventory_id": "INV-2001",
            "sku": "SKU-WIDGET-100",
            "description": "Standard Widget",
            "quantity": 5,
            "warehouse_location": "Aisle B, Bin 12",
            "comments": "Low stock alert.",
        },
    },
    "customs_declarations": {
        "CD-3001": {
            "declaration_id": "CD-3001",
            "shipment_id": "SH-1001",
            "hs_code": "8471.30",
            "declared_value": 15000.0,
            "additional_info": "Electronic components",
            "status": "pending",
        },
    },
    "carrier_messages": {
        "CM-4001": {
            "message_id": "CM-4001",
            "from": "carrier@shipping.example",
            "to": "ops@logistics.example",
            "subject": "Delay notice",
            "body": "Shipment SH-1001 delayed by weather.",
            "shipment_id": "SH-1001",
            "timestamp": "2025-01-20",
        },
    },
    "delivery_schedules": {
        "DS-5001": {
            "schedule_id": "DS-5001",
            "shipment_id": "SH-1001",
            "delivery_window_start": "2025-02-01 08:00",
            "delivery_window_end": "2025-02-01 12:00",
            "special_instructions": "Dock 3 only.",
            "confirmed": True,
        },
    },
    "current_user": {
        "operator_id": "OP-1",
        "name": "Lisa Logistics",
        "email": "lisa@logistics.example",
        "role": "dispatcher",
        "clearance_level": "standard",
    },
}


class TestGetCarrierMessages:
    def test_all(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_carrier_messages(env)
        assert len(results) == 1
        assert results[0]["message_id"] == "CM-4001"

    def test_by_shipment(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_carrier_messages(env, shipment_id="SH-1001")
        assert len(results) == 1
        results = get_carrier_messages(env, shipment_id="SH-9999")
        assert results == []


class TestSendCarrierMessage:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = send_carrier_message(
            env, to="carrier@shipping.example", subject="ETA update", body="Please confirm ETA."
        )
        assert "sent" in result
        assert len(env["carrier_messages"]) == 2


class TestGetCustomsDeclarationByShipment:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        decl = get_customs_declaration_by_shipment(env, shipment_id="SH-1001")
        assert decl["hs_code"] == "8471.30"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_customs_declaration_by_shipment(env, shipment_id="SH-9999")
        assert "error" in result


class TestSubmitCustomsDeclaration:
    def test_update_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        decl = submit_customs_declaration(env, shipment_id="SH-1001", hs_code="8471.50", declared_value=20000.0)
        assert decl["hs_code"] == "8471.50"
        assert decl["declared_value"] == 20000.0
        assert decl["status"] == "submitted"

    def test_create_new(self):
        env = copy.deepcopy(SAMPLE_ENV)
        decl = submit_customs_declaration(
            env, shipment_id="SH-NEW", hs_code="6109.10", declared_value=500.0, additional_info="Textiles"
        )
        assert decl["shipment_id"] == "SH-NEW"
        assert decl["status"] == "submitted"
        assert len(env["customs_declarations"]) == 2


class TestGetDeliverySchedule:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        schedule = get_delivery_schedule(env, shipment_id="SH-1001")
        assert schedule["schedule_id"] == "DS-5001"
        assert schedule["confirmed"] is True

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_delivery_schedule(env, shipment_id="SH-9999")
        assert "error" in result


class TestGetShipmentById:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        shipment = get_shipment_by_id(env, shipment_id="SH-1001")
        assert shipment["tracking_number"] == "TRK-99887"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_shipment_by_id(env, shipment_id="SH-9999")
        assert "error" in result


class TestSearchShipmentsByStatus:
    def test_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_shipments_by_status(env, status="in_transit")
        assert len(results) == 1
        assert results[0]["shipment_id"] == "SH-1001"

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_shipments_by_status(env, status="delivered")
        assert results == []


class TestUpdateShipmentStatus:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_shipment_status(env, shipment_id="SH-1001", status="delivered", eta="2025-02-01")
        assert "updated" in result
        assert env["shipments"]["SH-1001"]["status"] == "delivered"

    def test_update_notes(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_shipment_status(env, shipment_id="SH-1001", status="delayed", notes="Weather delay.")
        assert "updated" in result
        assert env["shipments"]["SH-1001"]["notes"] == "Weather delay."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_shipment_status(env, shipment_id="SH-9999", status="delivered")
        assert "Error" in result


class TestGetWarehouseInventoryItem:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        item = get_warehouse_inventory_item(env, inventory_id="INV-2001")
        assert item["sku"] == "SKU-WIDGET-100"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_warehouse_inventory_item(env, inventory_id="INV-9999")
        assert "error" in result


class TestListLowStockItems:
    def test_below_threshold(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_low_stock_items(env, threshold=10)
        assert len(results) == 1
        assert results[0]["inventory_id"] == "INV-2001"

    def test_above_threshold(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_low_stock_items(env, threshold=3)
        assert results == []


class TestAddWarehouseInventory:
    def test_add(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_warehouse_inventory(
            env, sku="SKU-GADGET-200", description="Gadget", quantity=50, warehouse_location="Aisle A, Bin 1"
        )
        assert "added" in result
        assert len(env["warehouse_inventories"]) == 2


class TestAdjustWarehouseQuantity:
    def test_increase(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = adjust_warehouse_quantity(env, inventory_id="INV-2001", adjustment=10)
        assert "adjusted" in result
        assert env["warehouse_inventories"]["INV-2001"]["quantity"] == 15

    def test_decrease(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = adjust_warehouse_quantity(env, inventory_id="INV-2001", adjustment=-3)
        assert "adjusted" in result
        assert env["warehouse_inventories"]["INV-2001"]["quantity"] == 2

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = adjust_warehouse_quantity(env, inventory_id="INV-9999", adjustment=5)
        assert "Error" in result


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "get_carrier_messages",
            "send_carrier_message",
            "get_customs_declaration_by_shipment",
            "submit_customs_declaration",
            "get_delivery_schedule",
            "get_shipment_by_id",
            "search_shipments_by_status",
            "update_shipment_status",
            "get_warehouse_inventory_item",
            "list_low_stock_items",
            "add_warehouse_inventory",
            "adjust_warehouse_quantity",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
