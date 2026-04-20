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

from resources_servers.indirect_prompt_injection.healthcare_tools import (
    TOOL_HANDLERS,
    cancel_appointment,
    get_lab_results,
    get_messages,
    get_patient_record,
    get_prescriptions,
    get_schedule,
    prescribe_medication,
    schedule_appointment,
    search_patients,
    send_message,
    send_referral,
    update_patient_chart,
)


SAMPLE_ENV = {
    "patients": {
        "P001": {
            "patient_id": "P001",
            "name": "Maria Garcia",
            "dob": "1975-04-12",
            "ssn": "123-45-6789",
            "conditions": ["Type 2 Diabetes", "Hypertension"],
            "medications": ["Metformin 500mg"],
            "chart_notes": "Initial visit notes.",
        },
        "P002": {
            "patient_id": "P002",
            "name": "James Wilson",
            "dob": "1982-09-30",
            "ssn": "987-65-4321",
            "conditions": ["Asthma"],
            "medications": ["Albuterol Inhaler PRN"],
            "chart_notes": "",
        },
    },
    "appointments": {
        "A001": {
            "appointment_id": "A001",
            "patient_id": "P001",
            "date": "2024-01-15",
            "time": "09:30",
            "reason": "Diabetes follow-up",
            "status": "scheduled",
        },
        "A002": {
            "appointment_id": "A002",
            "patient_id": "P002",
            "date": "2024-01-16",
            "time": "14:00",
            "reason": "Asthma check",
            "status": "scheduled",
        },
    },
    "messages": [
        {
            "from": "nurse@clinic.example",
            "to": "doctor@clinic.example",
            "subject": "Lab results for P001",
            "body": "Please review the latest HbA1c.",
            "timestamp": "2024-01-14T08:30:00Z",
        }
    ],
    "prescriptions": {
        "P001": [{"medication": "Metformin", "dosage": "500mg", "instructions": "Twice daily with meals"}]
    },
    "lab_results": {"P001": [{"test": "HbA1c", "result": "7.8%", "comments": "Slightly elevated."}]},
    "current_user": {"name": "Dr. Smith", "role": "physician", "email": "drsmith@clinic.example"},
}


class TestSearchPatients:
    def test_search_by_name(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_patients(env, query="Maria")
        assert len(results) == 1
        assert results[0]["patient_id"] == "P001"

    def test_search_by_condition(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_patients(env, query="Asthma")
        assert len(results) == 1
        assert results[0]["patient_id"] == "P002"

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_patients(env, query="nonexistent")
        assert results == []

    def test_search_case_insensitive(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_patients(env, query="maria")
        assert len(results) == 1


class TestGetPatientRecord:
    def test_existing_patient(self):
        env = copy.deepcopy(SAMPLE_ENV)
        record = get_patient_record(env, patient_id="P001")
        assert record["name"] == "Maria Garcia"

    def test_missing_patient(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_patient_record(env, patient_id="P999")
        assert "error" in result


class TestUpdatePatientChart:
    def test_append_notes(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_patient_chart(env, patient_id="P001", notes="New lab results reviewed.")
        assert "Chart updated" in result
        assert "New lab results reviewed." in env["patients"]["P001"]["chart_notes"]

    def test_missing_patient(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_patient_chart(env, patient_id="P999", notes="test")
        assert "Error" in result


class TestGetSchedule:
    def test_existing_date(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_schedule(env, date="2024-01-15")
        assert len(results) == 1
        assert results[0]["patient_id"] == "P001"

    def test_no_appointments(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_schedule(env, date="2099-01-01")
        assert results == []


class TestScheduleAppointment:
    def test_schedule(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = schedule_appointment(env, patient_id="P001", date="2024-02-01", time="10:00", reason="Checkup")
        assert "scheduled" in result
        assert len(env["appointments"]) == 3


class TestCancelAppointment:
    def test_cancel_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = cancel_appointment(env, appointment_id="A001")
        assert "cancelled" in result
        assert env["appointments"]["A001"]["status"] == "cancelled"

    def test_cancel_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = cancel_appointment(env, appointment_id="A999")
        assert "Error" in result


class TestSendReferral:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = send_referral(env, patient_id="P001", specialist_email="cardio@hospital.com", reason="Heart concern")
        assert "Referral sent" in result
        assert len(env["sent_actions"]) == 1
        assert env["sent_actions"][0]["specialist_email"] == "cardio@hospital.com"


class TestSendMessage:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        initial_count = len(env["messages"])
        result = send_message(env, recipient="admin@clinic.example", subject="Test", body="Hello")
        assert "Message sent" in result
        assert len(env["messages"]) == initial_count + 1
        assert env["messages"][-1]["to"] == "admin@clinic.example"


class TestGetMessages:
    def test_get(self):
        env = copy.deepcopy(SAMPLE_ENV)
        messages = get_messages(env)
        assert len(messages) == 1
        assert messages[0]["subject"] == "Lab results for P001"


class TestPrescribeMedication:
    def test_prescribe(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = prescribe_medication(
            env, patient_id="P001", medication="Insulin", dosage="10 units", instructions="Before meals"
        )
        assert "Prescribed" in result
        assert len(env["prescriptions"]["P001"]) == 2

    def test_prescribe_new_patient(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = prescribe_medication(
            env, patient_id="P002", medication="Prednisone", dosage="5mg", instructions="Daily"
        )
        assert "Prescribed" in result
        assert len(env["prescriptions"]["P002"]) == 1


class TestGetPrescriptions:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_prescriptions(env, patient_id="P001")
        assert len(results) == 1

    def test_no_prescriptions(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_prescriptions(env, patient_id="P002")
        assert results == []


class TestGetLabResults:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_lab_results(env, patient_id="P001")
        assert len(results) == 1
        assert results[0]["test"] == "HbA1c"

    def test_no_results(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_lab_results(env, patient_id="P002")
        assert results == []


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "search_patients",
            "get_patient_record",
            "update_patient_chart",
            "get_schedule",
            "schedule_appointment",
            "cancel_appointment",
            "send_referral",
            "send_message",
            "get_messages",
            "prescribe_medication",
            "get_prescriptions",
            "get_lab_results",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
