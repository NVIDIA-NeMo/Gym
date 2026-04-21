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

from resources_servers.indirect_prompt_injection.insurance_tools import (
    TOOL_HANDLERS,
    add_adjuster_note,
    create_claim,
    flag_fraud_investigation,
    get_adjuster_notes_for_claim,
    get_claim_by_id,
    get_messages_for_claim,
    get_policy_by_id,
    process_payout,
    search_policies_by_holder,
    send_message,
    update_claim_description,
    update_policy_notes,
)


SAMPLE_ENV = {
    "policies": {
        "POL-1001": {
            "policy_id": "POL-1001",
            "policyholder_name": "Jane Doe",
            "policy_number": "PN-12345",
            "effective_date": "2024-01-01",
            "expiration_date": "2025-01-01",
            "coverage_type": "auto",
            "premium_amount": 1200.0,
            "status": "active",
            "notes": "Preferred customer.",
        },
    },
    "claims": {
        "CL-2001": {
            "claim_id": "CL-2001",
            "policy_id": "POL-1001",
            "claim_number": "CN-555",
            "date_of_loss": "2024-06-15",
            "description": "Fender bender on highway.",
            "claim_type": "collision",
            "amount_requested": 5000.0,
            "status": "open",
            "adjuster_id": "ADJ-100",
        },
    },
    "adjuster_notes": {
        "AN-3001": {
            "note_id": "AN-3001",
            "claim_id": "CL-2001",
            "adjuster_id": "ADJ-100",
            "note_text": "Initial inspection completed.",
            "timestamp": "2024-06-16",
        },
    },
    "messages": [
        {
            "from": "system@insurance.example",
            "to": "jane@example.com",
            "subject": "Claim update",
            "body": "Your claim is being reviewed.",
            "related_claim_id": "CL-2001",
        },
    ],
    "payouts": {},
    "fraud_investigation_flags": {},
    "current_user": {
        "user_id": "USR-1",
        "name": "Agent Brown",
        "email": "brown@insurance.example",
        "department": "Claims",
    },
}


class TestAddAdjusterNote:
    def test_add_note(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = add_adjuster_note(env, claim_id="CL-2001", adjuster_id="ADJ-100", note_text="Damage confirmed.")
        assert result["claim_id"] == "CL-2001"
        assert result["adjuster_id"] == "ADJ-100"
        assert result["note_text"] == "Damage confirmed."
        assert len(env["adjuster_notes"]) == 2


class TestGetAdjusterNotesForClaim:
    def test_existing_claim(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_adjuster_notes_for_claim(env, claim_id="CL-2001")
        assert len(results) == 1
        assert results[0]["note_id"] == "AN-3001"

    def test_no_notes(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_adjuster_notes_for_claim(env, claim_id="CL-9999")
        assert results == []


class TestCreateClaim:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_claim(
            env,
            policy_id="POL-1001",
            claim_number="CN-777",
            date_of_loss="2024-07-01",
            description="Hail damage.",
            claim_type="comprehensive",
            amount_requested=3000.0,
            adjuster_id="ADJ-200",
        )
        assert result["policy_id"] == "POL-1001"
        assert result["status"] == "open"
        assert len(env["claims"]) == 2


class TestGetClaimById:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_claim_by_id(env, claim_id="CL-2001")
        assert result["claim_number"] == "CN-555"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_claim_by_id(env, claim_id="CL-9999")
        assert "error" in result


class TestUpdateClaimDescription:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_claim_description(env, claim_id="CL-2001", description="Rear-end collision.")
        assert "updated" in result
        assert env["claims"]["CL-2001"]["description"] == "Rear-end collision."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_claim_description(env, claim_id="CL-9999", description="N/A")
        assert "Error" in result


class TestGetMessagesForClaim:
    def test_existing_claim(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_messages_for_claim(env, claim_id="CL-2001")
        assert len(results) == 1
        assert results[0]["subject"] == "Claim update"

    def test_no_messages(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_messages_for_claim(env, claim_id="CL-9999")
        assert results == []


class TestFlagFraudInvestigation:
    def test_flag(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = flag_fraud_investigation(
            env, claim_id="CL-2001", investigator_id="INV-1", flag_reason="Suspicious pattern.", severity="high"
        )
        assert result["claim_id"] == "CL-2001"
        assert result["severity"] == "high"
        assert result["status"] == "open"
        assert len(env["fraud_investigation_flags"]) == 1


class TestProcessPayout:
    def test_process(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = process_payout(
            env, claim_id="CL-2001", amount=4500.0, date="2024-07-10", method="check", notes="Final settlement."
        )
        assert result["claim_id"] == "CL-2001"
        assert result["amount"] == 4500.0
        assert result["status"] == "processed"
        assert len(env["payouts"]) == 1


class TestGetPolicyById:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_policy_by_id(env, policy_id="POL-1001")
        assert result["policyholder_name"] == "Jane Doe"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_policy_by_id(env, policy_id="POL-9999")
        assert "error" in result


class TestSearchPoliciesByHolder:
    def test_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_policies_by_holder(env, policyholder_name="Jane")
        assert len(results) == 1
        assert results[0]["policy_id"] == "POL-1001"

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_policies_by_holder(env, policyholder_name="nonexistent")
        assert results == []


class TestUpdatePolicyNotes:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_policy_notes(env, policy_id="POL-1001", notes="VIP customer.")
        assert "updated" in result
        assert env["policies"]["POL-1001"]["notes"] == "VIP customer."

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_policy_notes(env, policy_id="POL-9999", notes="N/A")
        assert "Error" in result


class TestSendMessage:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        initial = len(env["messages"])
        result = send_message(env, to="jane@example.com", subject="Follow-up", body="Please provide documents.")
        assert "sent" in result.lower()
        assert len(env["messages"]) == initial + 1


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "add_adjuster_note",
            "get_adjuster_notes_for_claim",
            "create_claim",
            "get_claim_by_id",
            "update_claim_description",
            "send_message",
            "get_messages_for_claim",
            "flag_fraud_investigation",
            "process_payout",
            "get_policy_by_id",
            "search_policies_by_holder",
            "update_policy_notes",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
