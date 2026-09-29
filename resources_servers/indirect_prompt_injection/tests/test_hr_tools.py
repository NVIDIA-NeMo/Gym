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

from resources_servers.indirect_prompt_injection.hr_tools import (
    TOOL_HANDLERS,
    create_job_posting,
    get_candidate_profile,
    get_inbox,
    get_interview_schedule,
    get_open_positions,
    schedule_interview,
    search_candidates,
    send_email,
    send_offer,
    submit_interview_feedback,
    update_candidate_status,
)


SAMPLE_ENV = {
    "candidates": {
        "C-001": {
            "candidate_id": "C-001",
            "name": "Jordan Lee",
            "email": "jordan@email.com",
            "position_applied": "Data Analyst",
            "application_notes": "Strong SQL skills.",
            "status": "active",
            "resume": "5 years data analysis experience",
        },
        "C-002": {
            "candidate_id": "C-002",
            "name": "Alex Chen",
            "email": "alex@email.com",
            "position_applied": "Software Engineer",
            "application_notes": "Full-stack developer.",
            "status": "interviewing",
            "resume": "3 years Python experience",
        },
    },
    "interviews": {
        "I-001": {
            "interview_id": "I-001",
            "candidate_id": "C-002",
            "interviewer_email": "manager@company.com",
            "date": "2025-03-20",
            "time": "14:00",
            "status": "scheduled",
        },
    },
    "messages": [
        {
            "from": "hr@company.com",
            "to": "manager@company.com",
            "subject": "New applicant",
            "body": "Jordan Lee applied for Data Analyst.",
            "timestamp": "2025-03-15T10:00:00Z",
        }
    ],
    "open_positions": {
        "POS-001": {
            "position_id": "POS-001",
            "title": "Data Analyst",
            "description": "Analyze business data.",
            "requirements": "SQL, Python, 3+ years",
            "status": "open",
        }
    },
    "current_user": {"name": "Sarah Miller", "role": "recruiter", "email": "sarah@company.com"},
}


class TestSearchCandidates:
    def test_search_by_name(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_candidates(env, query="Jordan")
        assert len(results) == 1
        assert results[0]["candidate_id"] == "C-001"

    def test_search_by_skill(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_candidates(env, query="Python")
        assert len(results) == 1
        assert results[0]["candidate_id"] == "C-002"

    def test_search_with_status_filter(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_candidates(env, query="Jordan", status="active")
        assert len(results) == 1
        results = search_candidates(env, query="Jordan", status="interviewing")
        assert len(results) == 0

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_candidates(env, query="nonexistent")
        assert results == []


class TestGetCandidateProfile:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        profile = get_candidate_profile(env, candidate_id="C-001")
        assert profile["name"] == "Jordan Lee"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_candidate_profile(env, candidate_id="C-999")
        assert "error" in result


class TestUpdateCandidateStatus:
    def test_update(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_candidate_status(env, candidate_id="C-001", status="interviewing")
        assert "updated" in result
        assert env["candidates"]["C-001"]["status"] == "interviewing"

    def test_missing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_candidate_status(env, candidate_id="C-999", status="rejected")
        assert "Error" in result


class TestScheduleInterview:
    def test_schedule(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = schedule_interview(
            env, candidate_id="C-001", interviewer_email="tech@company.com", date="2025-04-01", time="10:00"
        )
        assert "scheduled" in result
        assert len(env["interviews"]) == 2


class TestGetInterviewSchedule:
    def test_all(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_interview_schedule(env)
        assert len(results) == 1

    def test_by_date(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_interview_schedule(env, date="2025-03-20")
        assert len(results) == 1
        results = get_interview_schedule(env, date="2099-01-01")
        assert results == []


class TestSubmitInterviewFeedback:
    def test_existing_interview(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = submit_interview_feedback(env, candidate_id="C-002", feedback="Strong technical skills.", rating=4)
        assert "submitted" in result
        assert env["interviews"]["I-001"]["feedback"] == "Strong technical skills."
        assert env["interviews"]["I-001"]["rating"] == 4

    def test_no_existing_interview(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = submit_interview_feedback(env, candidate_id="C-001", feedback="Good communicator.", rating=3)
        assert "submitted" in result
        assert len(env["interviews"]) == 2


class TestSendOffer:
    def test_send(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = send_offer(env, candidate_id="C-001", position="Data Analyst", salary="$80k", start_date="2025-05-01")
        assert "Offer sent" in result
        assert len(env["sent_actions"]) == 1
        assert env["sent_actions"][0]["candidate_id"] == "C-001"


class TestSendEmail:
    def test_send_with_messages(self):
        env = copy.deepcopy(SAMPLE_ENV)
        initial = len(env["messages"])
        result = send_email(env, recipient="manager@company.com", subject="Update", body="Status update.")
        assert "Email sent" in result
        assert len(env["messages"]) == initial + 1

    def test_send_with_inbox(self):
        env = copy.deepcopy(SAMPLE_ENV)
        env["inbox"] = env.pop("messages")
        initial = len(env["inbox"])
        result = send_email(env, recipient="manager@company.com", subject="Test", body="Hello.")
        assert "Email sent" in result
        assert len(env["inbox"]) == initial + 1


class TestGetInbox:
    def test_from_messages(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_inbox(env)
        assert len(results) == 1

    def test_from_inbox_key(self):
        env = copy.deepcopy(SAMPLE_ENV)
        env["inbox"] = env.pop("messages")
        results = get_inbox(env)
        assert len(results) == 1


class TestGetOpenPositions:
    def test_get(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_open_positions(env)
        assert len(results) == 1
        assert results[0]["title"] == "Data Analyst"


class TestCreateJobPosting:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_job_posting(
            env, title="Backend Engineer", description="Build APIs.", requirements="Go, 2+ years"
        )
        assert "created" in result
        assert len(env["open_positions"]) == 2


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "search_candidates",
            "get_candidate_profile",
            "update_candidate_status",
            "schedule_interview",
            "get_interview_schedule",
            "submit_interview_feedback",
            "send_offer",
            "send_email",
            "get_inbox",
            "get_open_positions",
            "create_job_posting",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
