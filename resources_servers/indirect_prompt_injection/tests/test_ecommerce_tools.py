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

from resources_servers.indirect_prompt_injection.ecommerce_tools import (
    TOOL_HANDLERS,
    add_order_note,
    create_support_ticket,
    get_order_details,
    get_product_details,
    get_ticket_details,
    process_refund,
    search_orders,
    search_products,
    update_order_status,
    update_product_stock,
    update_ticket_status,
)


SAMPLE_ENV = {
    "orders": {
        "O-1001": {
            "order_id": "O-1001",
            "customer_id": "CUST-100",
            "order_date": "2025-01-15",
            "status": "pending",
            "total_amount": 99.99,
            "shipping_address": "123 Main St",
            "notes": "",
        },
    },
    "products": {
        "P-2001": {
            "product_id": "P-2001",
            "name": "Wireless Mouse",
            "description": "Ergonomic wireless mouse",
            "price": 29.99,
            "category": "electronics",
            "stock_quantity": 50,
            "tags": ["peripherals", "wireless"],
        },
    },
    "support_tickets": {
        "T-3001": {
            "ticket_id": "T-3001",
            "customer_id": "CUST-100",
            "subject": "Defective item",
            "description": "Mouse stopped working",
            "status": "open",
            "priority": "medium",
        },
    },
    "refunds": {
        "R-4001": {
            "refund_id": "R-4001",
            "order_id": "O-1001",
            "customer_id": "CUST-100",
            "amount": 0,
            "status": "requested",
        },
    },
    "current_user": {"agent_id": "AGT-1", "name": "Agent Smith", "email": "smith@store.example"},
}


class TestSearchOrders:
    def test_search_by_customer_id(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_orders(env, customer_id="CUST-100")
        assert len(results) == 1
        assert results[0]["order_id"] == "O-1001"

    def test_search_by_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_orders(env, status="pending")
        assert len(results) == 1
        assert results[0]["order_id"] == "O-1001"

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_orders(env, customer_id="CUST-999")
        assert results == []


class TestGetOrderDetails:
    def test_existing_order(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_order_details(env, order_id="O-1001")
        assert result["customer_id"] == "CUST-100"
        assert result["total_amount"] == 99.99

    def test_missing_order(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_order_details(env, order_id="O-9999")
        assert "error" in result


class TestUpdateOrderStatus:
    def test_update_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_order_status(env, order_id="O-1001", status="shipped")
        assert "updated" in result
        assert env["orders"]["O-1001"]["status"] == "shipped"


class TestAddOrderNote:
    def test_add_note(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_order_note(env, order_id="O-1001", note="Customer called about delivery.")
        assert "Note added" in result
        assert "Customer called about delivery." in env["orders"]["O-1001"]["notes"]


class TestSearchProducts:
    def test_search_by_query(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_products(env, query="wireless")
        assert len(results) == 1
        assert results[0]["product_id"] == "P-2001"

    def test_search_by_category(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_products(env, category="electronics")
        assert len(results) == 1
        assert results[0]["product_id"] == "P-2001"


class TestGetProductDetails:
    def test_existing_product(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_product_details(env, product_id="P-2001")
        assert result["name"] == "Wireless Mouse"

    def test_missing_product(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_product_details(env, product_id="P-9999")
        assert "error" in result


class TestUpdateProductStock:
    def test_update_stock(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_product_stock(env, product_id="P-2001", stock_quantity=100)
        assert "Stock updated" in result
        assert env["products"]["P-2001"]["stock_quantity"] == 100


class TestProcessRefund:
    def test_existing_refund(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = process_refund(env, refund_id="R-4001", amount=49.99, reason="Defective")
        assert "processed" in result
        assert env["refunds"]["R-4001"]["status"] == "processed"
        assert env["refunds"]["R-4001"]["amount"] == 49.99

    def test_new_refund(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = process_refund(env, refund_id="R-NEW", amount=25.00, reason="Changed mind")
        assert "processed" in result
        assert "R-NEW" in env["refunds"]
        assert env["refunds"]["R-NEW"]["amount"] == 25.00


class TestCreateSupportTicket:
    def test_create_ticket(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_support_ticket(
            env, customer_id="CUST-100", subject="Shipping delay", description="Order not arrived"
        )
        assert "created" in result
        assert len(env["support_tickets"]) == 2


class TestGetTicketDetails:
    def test_existing_ticket(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_ticket_details(env, ticket_id="T-3001")
        assert result["subject"] == "Defective item"

    def test_missing_ticket(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_ticket_details(env, ticket_id="T-9999")
        assert "error" in result


class TestUpdateTicketStatus:
    def test_update_status(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_ticket_status(env, ticket_id="T-3001", status="resolved")
        assert "updated" in result
        assert env["support_tickets"]["T-3001"]["status"] == "resolved"


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "search_orders",
            "get_order_details",
            "update_order_status",
            "add_order_note",
            "search_products",
            "get_product_details",
            "update_product_stock",
            "process_refund",
            "create_support_ticket",
            "get_ticket_details",
            "update_ticket_status",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
