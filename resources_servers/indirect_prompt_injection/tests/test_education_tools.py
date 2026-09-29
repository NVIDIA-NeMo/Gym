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

from resources_servers.indirect_prompt_injection.education_tools import (
    TOOL_HANDLERS,
    adjust_financial_aid,
    create_course,
    get_course_detail,
    get_financial_aid,
    get_grades_for_student,
    get_student_record,
    list_courses,
    search_students,
    submit_grade,
    update_student_advisor_notes,
)


SAMPLE_ENV = {
    "students": {
        "S-1001": {
            "student_id": "S-1001",
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice@university.edu",
            "major": "Computer Science",
            "year": 3,
            "gpa": 3.8,
            "advisor_notes": "Strong student.",
        },
    },
    "courses": {
        "CS101": {
            "course_id": "CS101",
            "course_code": "CS101",
            "title": "Intro to Programming",
            "description": "Learn basic programming",
            "credits": 3,
            "department": "Computer Science",
            "instructor_id": "PROF-1",
            "schedule": "MWF 10:00-11:00",
        },
    },
    "grades": {
        "G-2001": {
            "grade_id": "G-2001",
            "student_id": "S-1001",
            "course_id": "CS101",
            "assignment_name": "Midterm",
            "score": 92,
            "max_score": 100,
            "grader_comments": "Excellent work.",
        },
    },
    "financial_aid": {
        "FA-3001": {
            "aid_id": "FA-3001",
            "student_id": "S-1001",
            "award_type": "scholarship",
            "amount": 5000,
            "academic_year": "2025",
            "status": "pending",
            "reviewer_notes": "",
        },
    },
    "messages": [],
    "current_user": {"staff_id": "STAFF-1", "name": "Dr. Smith", "email": "smith@university.edu", "department": "CS"},
}


class TestListCourses:
    def test_list_all(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_courses(env)
        assert len(results) == 1
        assert results[0]["course_id"] == "CS101"

    def test_list_by_department(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_courses(env, department="Computer Science")
        assert len(results) == 1
        assert results[0]["course_id"] == "CS101"

    def test_list_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = list_courses(env, department="History")
        assert results == []


class TestGetCourseDetail:
    def test_existing_course(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_course_detail(env, course_id="CS101")
        assert result["title"] == "Intro to Programming"

    def test_missing_course(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_course_detail(env, course_id="CS999")
        assert "error" in result


class TestCreateCourse:
    def test_create(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = create_course(
            env,
            course_code="CS201",
            title="Data Structures",
            description="Advanced data structures",
            credits=4,
            department="Computer Science",
            instructor_id="PROF-2",
            schedule="TTh 13:00-14:30",
        )
        assert "created" in result
        assert len(env["courses"]) == 2


class TestGetFinancialAid:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_financial_aid(env, student_id="S-1001")
        assert len(results) == 1
        assert results[0]["aid_id"] == "FA-3001"

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_financial_aid(env, student_id="S-9999")
        assert results == []


class TestAdjustFinancialAid:
    def test_adjust_all_fields(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = adjust_financial_aid(
            env, aid_id="FA-3001", amount=6000, status="approved", reviewer_notes="Increased award."
        )
        assert result["amount"] == 6000
        assert result["status"] == "approved"
        assert result["reviewer_notes"] == "Increased award."

    def test_missing_aid(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = adjust_financial_aid(env, aid_id="FA-9999", amount=1000)
        assert "error" in result


class TestGetGradesForStudent:
    def test_existing(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_grades_for_student(env, student_id="S-1001")
        assert len(results) == 1
        assert results[0]["assignment_name"] == "Midterm"

    def test_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = get_grades_for_student(env, student_id="S-9999")
        assert results == []


class TestSubmitGrade:
    def test_submit(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = submit_grade(
            env,
            student_id="S-1001",
            course_id="CS101",
            assignment_name="Final",
            score=88,
            max_score=100,
            grader_comments="Good effort.",
        )
        assert "submitted" in result
        assert len(env["grades"]) == 2


class TestSearchStudents:
    def test_search_by_name(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_students(env, name="Alice")
        assert len(results) == 1
        assert results[0]["student_id"] == "S-1001"

    def test_search_by_major(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_students(env, major="Computer Science")
        assert len(results) == 1
        assert results[0]["student_id"] == "S-1001"

    def test_search_no_match(self):
        env = copy.deepcopy(SAMPLE_ENV)
        results = search_students(env, name="Nonexistent")
        assert results == []


class TestGetStudentRecord:
    def test_existing_student(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_student_record(env, student_id="S-1001")
        assert result["first_name"] == "Alice"

    def test_missing_student(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = get_student_record(env, student_id="S-9999")
        assert "error" in result


class TestUpdateStudentAdvisorNotes:
    def test_existing_student(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_student_advisor_notes(env, student_id="S-1001", notes="Recommended for TA position.")
        assert "updated" in result
        assert "Recommended for TA position." in env["students"]["S-1001"]["advisor_notes"]

    def test_missing_student(self):
        env = copy.deepcopy(SAMPLE_ENV)
        result = update_student_advisor_notes(env, student_id="S-9999", notes="test")
        assert "Error" in result


class TestToolHandlers:
    def test_all_tools_registered(self):
        expected = {
            "send_message",
            "get_messages",
            "list_courses",
            "get_course_detail",
            "create_course",
            "get_financial_aid",
            "adjust_financial_aid",
            "get_grades_for_student",
            "submit_grade",
            "search_students",
            "get_student_record",
            "update_student_advisor_notes",
        }
        assert set(TOOL_HANDLERS.keys()) == expected
