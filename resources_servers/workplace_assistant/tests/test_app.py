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
from unittest.mock import MagicMock, patch

import pytest


pd = pytest.importorskip("pandas")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient
from pytest import fixture

from nemo_gym.mcp_test_utils import (
    TOKEN_HEADER,
    assert_transport_parity,
    http_tool_names,
    mcp_call,
    mcp_handshake,
    mcp_list_tools,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.workplace_assistant.app import (
    TOOLKITS,
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    WorkbenchVerifyRequest,
)
from resources_servers.workplace_assistant.utils import get_tools


EXPECTED_TOOLS = frozenset(
    {
        "company_directory_find_email_address",
        "email_get_email_information_by_id",
        "email_search_emails",
        "email_send_email",
        "email_delete_email",
        "email_forward_email",
        "email_reply_email",
        "calendar_get_event_information_by_id",
        "calendar_search_events",
        "calendar_create_event",
        "calendar_delete_event",
        "calendar_update_event",
        "analytics_engaged_users_count",
        "analytics_get_visitor_information_by_id",
        "analytics_create_plot",
        "analytics_traffic_source_count",
        "analytics_total_visits_count",
        "analytics_get_average_session_duration",
        "project_management_get_task_information_by_id",
        "project_management_search_tasks",
        "project_management_create_task",
        "project_management_delete_task",
        "project_management_update_task",
        "customer_relationship_manager_search_customers",
        "customer_relationship_manager_update_customer",
        "customer_relationship_manager_add_customer",
        "customer_relationship_manager_delete_customer",
    }
)


class TestApp:
    @fixture
    def config(self) -> WorkbenchResourcesServerConfig:
        return WorkbenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    def init_server(self, config: WorkbenchResourcesServerConfig):
        server_mock = MagicMock(spec=ServerClient)
        resources_server = WorkbenchResourcesServer(config=config, server_client=server_mock)
        return resources_server

    def client(self, resources_server: WorkbenchResourcesServer) -> TestClient:
        return TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000")

    def seed(self, client: TestClient) -> None:
        resp = client.post("/seed_session", json={})
        assert resp.status_code == 200, resp.text

    def mock_analytics_reset_state(self, analytics_tool_instance):
        """
        This function will be used as the new reset_state.
        It sets the necessary attributes but keeps 'user_engaged' as a string.
        """
        analytics_tool_instance._analytics_data = pd.DataFrame(
            [{"user_engaged": "False"}]  # Keep as a string
        )
        analytics_tool_instance._plots_data = pd.DataFrame(columns=["file_path"])

    def test_company_directory_find_email_address(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = {"email_address": ["aisha.chen@atlas.com", "carlos.rodriguez@atlas.com"]}
        mock_df = pd.DataFrame(mock_data)
        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.company_directory.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/company_directory_find_email_address", json={"name": "aisha"})
                assert response.status_code == 200
                assert response.json() == {"output": ["aisha.chen@atlas.com"]}

    def test_email_get_email_information_by_id(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = {
            "email_id": ["00000393", "00000123"],
            "inbox/outbox": ["inbox", "outbox"],
            "sender/recipient": ["raj.patel@atlas.com", "test@example.com"],
            "subject": ["Update on Supply Chain", "Test Subject"],
            "sent_datetime": ["2023-10-01 13:11:52", "2023-10-01 14:00:00"],
            "body": ["Dear Sam, I have some ideas...", "This is a test body."],
        }
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/email_get_email_information_by_id", json={"email_id": "00000393", "field": "inbox/outbox"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"inbox/outbox": "inbox"}}

    def test_email_search_emails(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "email_id": "match_01",
                "inbox/outbox": "inbox",
                "sender/recipient": "aisha.chen@atlas.com",
                "subject": "Matching Email Subject",
                "sent_datetime": "2025-09-01 10:00:00",
                "body": "This is a test email from Aisha.",
            },
            {
                "email_id": "no_match_01",
                "inbox/outbox": "inbox",
                "sender/recipient": "another.user@example.com",
                "subject": "Non-Matching Subject",
                "sent_datetime": "2025-09-01 11:00:00",
                "body": "This email should not be found.",
            },
        ]
        mock_df = pd.DataFrame(mock_data)
        expected_output = {
            "emails": [mock_data[0]],
            "pagination": {
                "total_emails": 1,
                "page": 1,
                "page_size": 5,
                "total_pages": 1,
            },
        }

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/email_search_emails", json={"query": "aisha.chen@atlas.com"})
                assert response.status_code == 200
                assert response.json() == {"output": expected_output}

    def test_email_send_email(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "email_id": "123",
                "inbox/outbox": "inbox",
                "sender/recipient": "initial.user@example.com",
                "subject": "Initial Email",
                "sent_datetime": "2025-01-01 12:00:00",
                "body": "This is a pre-existing email.",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/email_send_email",
                    json={
                        "recipient": "aisha.chen@atlas.com",
                        "subject": "regarding something",
                        "body": "some body",
                    },
                )
                assert response.status_code == 200
                # Byte-equal replay of the historical wire contract.
                assert response.content == b'{"output":"Email sent successfully."}'

    def test_email_delete_email(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "email_id": "00000393",
                "inbox/outbox": "inbox",
                "sender/recipient": "user.one@example.com",
                "subject": "Email to be deleted",
                "sent_datetime": "2025-01-01 12:00:00",
                "body": "This is a test email.",
            },
            {
                "email_id": "00000123",
                "inbox/outbox": "inbox",
                "sender/recipient": "user.two@example.com",
                "subject": "Another Email",
                "sent_datetime": "2025-01-01 12:05:00",
                "body": "This is another test email.",
            },
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/email_delete_email", json={"email_id": "00000393"})
                assert response.status_code == 200
                assert response.json() == {"output": "Email deleted successfully."}

    def test_email_forward_email(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "email_id": "00000393",
                "inbox/outbox": "inbox",
                "sender/recipient": "original.sender@example.com",
                "subject": "Original Subject",
                "sent_datetime": "2025-01-01 12:00:00",
                "body": "This is the original email body.",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/email_forward_email", json={"email_id": "00000393", "recipient": "aisha.chen@atlas.com"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": "Email forwarded successfully."}

    def test_calendar_get_event_information_by_id(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "event_id": "00000013",
                "event_name": "Test Event",
                "participant_email": "test.user@example.com",
                "event_start": "2025-09-02 10:00:00",
                "duration": "60",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.calendar.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/calendar_get_event_information_by_id", json={"event_id": "00000013", "field": "event_id"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"event_id": "00000013"}}

    def test_calendar_search_events(self, config: WorkbenchResourcesServerConfig) -> None:
        entry_1 = {
            "event_id": "00000016",
            "event_name": "sync up",
            "participant_email": "santiago.martinez@atlas.com",
            "event_start": "2023-12-12 12:00:00",
            "duration": "90",
        }
        entry_2 = {
            "event_id": "00000017",
            "event_name": "Team sync up",
            "participant_email": "aisha.chen@atlas.com",
            "event_start": "2023-12-13 10:00:00",
            "duration": "30",
        }
        non_matching_entry = {
            "event_id": "00000018",
            "event_name": "Project Deadline",
            "participant_email": "sam@example.com",
            "event_start": "2023-12-14 17:00:00",
            "duration": "5",
        }
        mock_data = [entry_1, entry_2, non_matching_entry]
        mock_df = pd.DataFrame(mock_data)

        expected_output = {
            "events": [entry_2, entry_1],
            "pagination": {
                "total_events": 2,
                "page": 1,
                "page_size": 5,
                "total_pages": 1,
            },
        }

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.calendar.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/calendar_search_events", json={"query": "sync up"})
                assert response.status_code == 200
                assert response.json() == {"output": expected_output}

    def test_calendar_delete_event(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {"event_id": "00000013", "event_name": "Event to Delete"},
            {"event_id": "00000014", "event_name": "Another Event"},
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.calendar.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/calendar_delete_event", json={"event_id": "00000013"})
                assert response.status_code == 200
                assert response.json() == {"output": "Event deleted successfully."}

    def test_calendar_update_event(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "event_id": "00000013",
                "event_name": "Initial Name",
                "participant_email": "test@example.com",
                "event_start": "2025-01-01 10:00:00",
                "duration": "60",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.calendar.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/calendar_update_event", json={"event_id": "00000013", "field": "duration", "new_value": "100"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": "Event updated successfully."}

    def test_analytics_engaged_users_count(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "visitor_id": "0860",
                "page_views": 8,
                "session_duration_seconds": 4,
                "traffic_source": "referral",
                "user_engaged": False,
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/analytics_engaged_users_count", json={"time_min": "2023-10-22", "time_max": "2023-11-22"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"2023-10-22": 0}}

    def test_analytics_get_visitor_information_by_id(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "visitor_id": "0860",
                "page_views": "8",
                "session_duration_seconds": "4",
                "traffic_source": "referral",
                "user_engaged": "False",
            },
            {
                "date_of_visit": "2023-11-22",
                "visitor_id": "4426",
                "page_views": "3",
                "session_duration_seconds": "12",
                "traffic_source": "direct",
                "user_engaged": "True",
            },
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/analytics_get_visitor_information_by_id", json={"visitor_id": "0860"})
                assert response.status_code == 200
                assert response.json() == {
                    "output": [
                        {
                            "date_of_visit": "2023-10-22",
                            "visitor_id": "0860",
                            "page_views": "8",
                            "session_duration_seconds": "4",
                            "traffic_source": "referral",
                            "user_engaged": False,
                        }
                    ]
                }

    def test_analytics_create_plot(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "page_views": "10",
                "user_engaged": "False",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/analytics_create_plot",
                    json={
                        "time_min": "2023-10-22",
                        "time_max": "2023-10-22",
                        "value_to_plot": "total_visits",
                        "plot_type": "bar",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": "plots/2023-10-22_2023-10-22_total_visits_bar.png"}

    def test_analytics_traffic_source_count(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "traffic_source": "referral",
                "user_engaged": "False",
            },
            {
                "date_of_visit": "2023-10-22",
                "traffic_source": "referral",
                "user_engaged": "False",
            },
            {
                "date_of_visit": "2023-10-22",
                "traffic_source": "direct",
                "user_engaged": "False",
            },
            {
                "date_of_visit": "2023-10-23",
                "traffic_source": "referral",
                "user_engaged": "False",
            },
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/analytics_traffic_source_count",
                    json={
                        "time_min": "2023-10-22",
                        "time_max": "2023-10-22",
                        "traffic_source": "referral",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"2023-10-22": 2}}

    def test_analytics_total_visits_count(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"date_of_visit": "2023-10-22", "user_engaged": "False"} for _ in range(13)] + [
            {"date_of_visit": "2023-10-23", "user_engaged": "False"} for _ in range(2)
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/analytics_total_visits_count", json={"time_min": "2023-10-22", "time_max": "2023-10-22"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"2023-10-22": 13}}

    def test_analytics_get_average_session_duration(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = (
            [
                {
                    "date_of_visit": "2023-10-22",
                    "session_duration_seconds": "15",
                    "user_engaged": "False",
                }
                for _ in range(12)
            ]
            + [
                {
                    "date_of_visit": "2023-10-22",
                    "session_duration_seconds": "25",
                    "user_engaged": "False",
                }
            ]
            + [
                {
                    "date_of_visit": "2023-10-23",
                    "session_duration_seconds": "100",
                    "user_engaged": "False",
                }
            ]
        )
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.pd.read_csv"
            ) as mock_read_csv,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/analytics_get_average_session_duration",
                    json={"time_min": "2023-10-22", "time_max": "2023-10-22"},
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"2023-10-22": 15.76923076923077}}

    def test_project_management_get_task_information_by_id(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"task_id": "00000149", "task_name": "Test Task"}]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.project_management.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/project_management_get_task_information_by_id",
                    json={"task_id": "00000149", "field": "task_id"},
                )
                assert response.status_code == 200
                assert response.json() == {"output": {"task_id": "00000149"}}

    def test_project_management_search_tasks(self, config: WorkbenchResourcesServerConfig) -> None:
        expected_tasks = [
            {
                "task_id": "00000149",
                "task_name": "Add animation to carousel",
                "assigned_to_email": "leila.azizi@atlas.com",
                "list_name": "Backlog",
                "due_date": "2023-11-27",
                "board": "Front end",
            },
            {
                "task_id": "00000151",
                "task_name": "Fix alignment issue in profile page",
                "assigned_to_email": "leila.azizi@atlas.com",
                "list_name": "Backlog",
                "due_date": "2023-11-27",
                "board": "Front end",
            },
        ]
        non_matching_task = [
            {
                "task_id": "00000999",
                "task_name": "Irrelevant Task",
                "assigned_to_email": "other@user.com",
                "list_name": "Done",
                "due_date": "2025-01-01",
                "board": "Design",
            }
        ]
        mock_df = pd.DataFrame(expected_tasks + non_matching_task)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.project_management.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/project_management_search_tasks",
                    json={
                        "task_name": "",
                        "assigned_to_email": "leila.azizi@atlas.com",
                        "list_name": "Backlog",
                        "due_date": "2023-11-27",
                        "board": "Front end",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": expected_tasks}

    def test_project_management_create_task(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "task_id": "00000299",
                "assigned_to_email": "leila.azizi@atlas.com",
                "task_name": "Existing Task",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.project_management.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/project_management_create_task",
                    json={
                        "task_name": "Add animation to carousel",
                        "assigned_to_email": "leila.azizi@atlas.com",
                        "list_name": "Backlog",
                        "due_date": "2023-11-27",
                        "board": "Front end",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": "00000300"}

    def test_project_management_delete_task(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"task_id": "00000149", "task_name": "A task to delete"}]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.project_management.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post("/project_management_delete_task", json={"task_id": "00000149"})
                assert response.status_code == 200
                assert response.json() == {"output": "Task deleted successfully."}

    def test_project_management_update_task(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [
            {
                "task_id": "00000149",
                "task_name": "Some Task",
                "board": "Front end",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.project_management.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/project_management_update_task",
                    json={"task_id": "00000149", "field": "board", "new_value": "Design"},
                )
                assert response.status_code == 200
                assert response.json() == {"output": "Task updated successfully."}

    def test_customer_relationship_manager_search_customers(self, config: WorkbenchResourcesServerConfig) -> None:
        expected_customer = {
            "customer_id": "00000052",
            "assigned_to_email": "raj.patel@atlas.com",
            "customer_name": "Jaden White",
            "customer_email": "jaden.white@protracefoods",
            "customer_phone": "724-857-2625",
            "last_contact_date": "2023-11-30 23:59:00",
            "product_interest": "Hardware",
            "status": "Won",
            "follow_up_by": "2023-12-13 23:59:00",
            "notes": "2023-10-17: Had a call. ",
        }
        non_matching_customer = {"customer_id": "00000999", "customer_name": "Jane Doe"}
        mock_df = pd.DataFrame([expected_customer, non_matching_customer])

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/customer_relationship_manager_search_customers",
                    json={
                        "customer_name": "Jaden White",
                        "customer_email": "jaden.white@protracefoods",
                        "product_interest": "Hardware",
                        "status": "Won",
                        "assigned_to_email": "raj.patel@atlas.com",
                        "last_contact_date_min": "2023-11-30 23:59:00",
                        "last_contact_date_max": "2023-11-30 23:59:00",
                        "follow_up_by_min": "2023-12-13 23:5",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {
                    "output": {
                        "customers": [expected_customer],
                        "pagination": {
                            "total_customers": 1,
                            "page": 1,
                            "page_size": 5,
                            "total_pages": 1,
                        },
                    }
                }

    def test_customer_relationship_manager_update_customer(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"customer_id": "00000189", "customer_name": "old name"}]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/customer_relationship_manager_update_customer",
                    json={
                        "customer_id": "00000189",
                        "field": "customer_name",
                        "new_value": "new customer",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": "Customer updated successfully."}

    def test_customer_relationship_manager_add_customer(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"customer_id": "00000199", "customer_name": "Max ID Customer"}]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/customer_relationship_manager_add_customer",
                    json={
                        "customer_name": "Some customer",
                        "product_interest": "Hardware",
                        "status": "Won",
                        "assigned_to_email": "raj.patel@atlas.com",
                        "last_contact_date": "2023-11-30 23:59:00",
                        "follow_up_by": "2023-12-13 23:59:00",
                        "notes": "some notes",
                    },
                )
                assert response.status_code == 200
                assert response.json() == {"output": "00000200"}

    def test_customer_relationship_manager_delete_customer(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_data = [{"customer_id": "00000189", "customer_name": "Customer to Delete"}]
        mock_df = pd.DataFrame(mock_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.customer_relationship_manager.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/customer_relationship_manager_delete_customer", json={"customer_id": "00000189"}
                )
                assert response.status_code == 200
                assert response.json() == {"output": "Customer deleted successfully."}

    async def test_verify(self, config: WorkbenchResourcesServerConfig) -> None:
        mock_email_data = [
            {
                "email_id": "00000057",
                "inbox/outbox": "inbox",
                "sender/recipient": "carlos.rodriguez@atlas.com",
                "subject": "Task Update on Develop prototype for report generation",
                "sent_datetime": "2023-11-29 10:00:00",
                "body": "Just an update.",
            }
        ]
        mock_email_df = pd.DataFrame(mock_email_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_email_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_email_csv.return_value = mock_email_df

            resources_server = self.init_server(config)

            HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T23:59:00")
            SYS_PROMPT = (
                f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
                f"and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
                "Meetings must not start before 9am or end after 6pm."
            )

            responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {"role": "system", "content": SYS_PROMPT},
                    {
                        "role": "user",
                        "content": "Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "name": "email_reply_email",
                        "description": "Replies to an email by its ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "email_id": {
                                    "type": "string",
                                    "description": "Unique ID of the email to be replied",
                                },
                                "body": {
                                    "type": "string",
                                    "description": "Body content of the email",
                                },
                            },
                            "required": ["email_id", "body"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                    }
                ],
            )

            response = NeMoGymResponse(
                **{
                    "id": "resp_68b28c5cc7688195b56def0cfde4526a0527c59d17df88e4",
                    "created_at": 1756531804.0,
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "metadata": {},
                    "model": "gpt-4.1-2025-04-14",
                    "object": "response",
                    "output": [
                        {
                            "arguments": '{"email_id": "00000057", "body": "Thanks for the update - I will get back to you tomorrow."}',
                            "call_id": "call_4HOX5l7EBfGNHFUYwdzMu989",
                            "name": "email_reply_email",
                            "type": "function_call",
                            "id": "fc_68b2761bc78881909f6f5494de84736001454dbdc05b39d0",
                            "status": "completed",
                            "output": None,
                            "content": None,
                            "role": None,
                        }
                    ],
                    "parallel_tool_calls": False,
                    "temperature": 1.0,
                    "tool_choice": "auto",
                    "tools": [
                        {
                            "name": "email_reply_email",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "email_id": {
                                        "type": "string",
                                        "description": "Unique ID of the email to be replied",
                                    },
                                    "body": {
                                        "type": "string",
                                        "description": "Body content of the email",
                                    },
                                },
                                "required": ["email_id", "body"],
                                "additionalProperties": False,
                            },
                            "strict": False,
                            "type": "function",
                            "description": "Replies to an email by its ID.",
                        }
                    ],
                }
            )

            verify_request = WorkbenchVerifyRequest(
                responses_create_params=responses_create_params,
                response=response,
                ground_truth=[
                    {
                        "name": "email_reply_email",
                        "arguments": '{"email_id": "00000057", "body": "Thanks for the update - I will get back to you tomorrow."}',
                    }
                ],
                category="workplace_assistant_email",
                environment_name="workplace_assistant",
                id="0",
            )

            verification_response = await resources_server.verify(verify_request)

            assert verification_response.reward == 1.0

    async def test_stateful_email_deletion_and_fetch(self, config: WorkbenchResourcesServerConfig) -> None:
        """
        Tests the statefulness of the session by deleting an email and then
        attempting to fetch it, expecting it to not be found.
        """
        mock_email_data = [
            {
                "email_id": "00000146",
                "inbox/outbox": "inbox",
                "sender/recipient": "jane.doe@example.com",
                "subject": "Project Update",
                "sent_datetime": "2025-09-22 14:30:00",
                "body": "Here is the project update you requested.",
            }
        ]
        mock_email_df = pd.DataFrame(mock_email_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_email_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_email_csv.return_value = mock_email_df

            resources_server = self.init_server(config)

            # Define the user's request (responses_create_params)
            # The prompt asks to perform two actions in sequence.
            responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    {
                        "role": "user",
                        "content": "Delete email with email_id 00000146 and then get email information for id 00000146 with field email_id.",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "name": "email_delete_email",
                        "description": "Deletes an email by its ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "email_id": {
                                    "type": "string",
                                    "description": "Unique ID of the email to be deleted",
                                }
                            },
                            "required": ["email_id"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                    },
                    {
                        "type": "function",
                        "name": "email_get_email_information_by_id",
                        "description": "Retrieves specific details of an email by its ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "email_id": {
                                    "type": "string",
                                    "description": "Unique ID of the email",
                                },
                                "field": {
                                    "type": "string",
                                    "description": "Specific field to return. Available fields: 'email_id', 'inbox/outbox', 'sender/recipient', 'subject', 'sent_datetime', 'body'",
                                },
                            },
                            "required": ["email_id", "field"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                    },
                ],
            )

            # Construct the expected multi-step response from the model
            # This simulates the entire conversation, including the state change.

            response = NeMoGymResponse(
                **{
                    "id": "resp_b67653d1a560455ab79d942c01c84386",
                    "created_at": 1758649796.0,
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "metadata": None,
                    "model": "gpt-4.1-2025-04-14",
                    "object": "response",
                    "output": [
                        {
                            "arguments": '{"email_id":"00000146"}',
                            "call_id": "call_7MMdD7QRpDVpRInTVKZzJN8a",
                            "name": "email_delete_email",
                            "type": "function_call",
                            "id": "call_7MMdD7QRpDVpRInTVKZzJN8a",
                            "status": "completed",
                        },
                        {
                            "call_id": "call_7MMdD7QRpDVpRInTVKZzJN8a",
                            "output": '{"output":"Email deleted successfully."}',
                            "type": "function_call_output",
                            "id": None,
                            "status": None,
                        },
                        {
                            "arguments": '{"email_id":"00000146","field":"email_id"}',
                            "call_id": "call_FFRvdyjcwUD6BIGsoDOGFtdr",
                            "name": "email_get_email_information_by_id",
                            "type": "function_call",
                            "id": "call_FFRvdyjcwUD6BIGsoDOGFtdr",
                            "status": "completed",
                        },
                        {
                            "call_id": "call_FFRvdyjcwUD6BIGsoDOGFtdr",
                            "output": '{"output":"Email not found."}',
                            "type": "function_call_output",
                            "id": None,
                            "status": None,
                        },
                    ],
                    "parallel_tool_calls": False,
                    "temperature": None,
                    "tool_choice": "auto",
                    "tools": [
                        {
                            "type": "function",
                            "name": "email_delete_email",
                            "description": "Deletes an email by its ID.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "email_id": {
                                        "type": "string",
                                        "description": "Unique ID of the email to be deleted",
                                    }
                                },
                                "required": ["email_id"],
                                "additionalProperties": False,
                            },
                            "strict": False,
                        },
                        {
                            "type": "function",
                            "name": "email_get_email_information_by_id",
                            "description": "Retrieves specific details of an email by its ID.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "email_id": {
                                        "type": "string",
                                        "description": "Unique ID of the email",
                                    },
                                    "field": {
                                        "type": "string",
                                        "description": "Specific field to return. Available fields: 'email_id', 'inbox/outbox', 'sender/recipient', 'subject', 'sent_datetime', 'body'",
                                    },
                                },
                                "required": ["email_id", "field"],
                                "additionalProperties": False,
                            },
                            "strict": False,
                        },
                    ],
                }
            )

            ground_truth = [
                {
                    "name": "email_delete_email",
                    "arguments": '{"email_id": "00000146"}',
                },
                {
                    "name": "email_get_email_information_by_id",
                    "arguments": '{"email_id": "00000146", "field": "email_id"}',
                },
            ]

            verify_request = WorkbenchVerifyRequest(
                responses_create_params=responses_create_params,
                response=response,
                ground_truth=ground_truth,
                category="workplace_assistant_email",
                environment_name="workplace_assistant",
                id="1",
            )

            verification_response = await resources_server.verify(verify_request)

            # The reward should be 1.0 because the predicted function calls in our
            # crafted `response` object exactly match the `ground_truth`.
            assert verification_response.reward == 1.0, (
                f"Verification failed with reward {verification_response.reward}"
            )

    def test_extra_arguments_error_handling(self, config: WorkbenchResourcesServerConfig) -> None:
        """
        Tests if the server correctly handles a TypeError when a tool function is called with an unexpected keyword argument and hence will be added as part of model context
        """

        mock_email_data = [
            {
                "email_id": "00000057",
                "sender/recipient": "carlos.rodriguez@atlas.com",
                "subject": "Task Update",
                "body": "Just an update.",
            }
        ]
        mock_email_df = pd.DataFrame(mock_email_data)

        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.email.pd.read_csv"
            ) as mock_email_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self.mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_email_csv.return_value = mock_email_df

            resources_server = self.init_server(config)

            with self.client(resources_server) as client:
                self.seed(client)
                response = client.post(
                    "/email_get_email_information_by_id",
                    json={
                        "email_id": "00000057",
                        "field": "subject",
                        "useless_argument": "this should cause an error",  # This is the extra argument
                    },
                )
                assert response.status_code == 200
                assert (
                    "get_email_information_by_id': EmailTool.get_email_information_by_id() got an unexpected keyword argument 'useless_argument'"
                    in response.json()["output"]
                )


class TestWireContractErrorPaths:
    """The historical soft error contracts: seeded-session 400s and 200-with-error-string bodies."""

    def _server(self) -> WorkbenchResourcesServer:
        config = WorkbenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="workplace_assistant")
        return WorkbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _client(self, server: WorkbenchResourcesServer) -> TestClient:
        return TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000")

    def test_known_tool_without_seed_is_400(self) -> None:
        with self._client(self._server()) as client:
            resp = client.post("/email_send_email", json={"recipient": "a", "subject": "b", "body": "c"})
            assert resp.status_code == 400
            assert resp.json() == {"detail": "Session not initialized. Please call seed_session first."}

    def test_unknown_tool_without_seed_is_400(self) -> None:
        with self._client(self._server()) as client:
            resp = client.post("/made_up_tool", json={})
            assert resp.status_code == 400
            assert resp.json() == {"detail": "Session not initialized. Please call seed_session first."}

    def test_unknown_tool_when_seeded_is_200_soft_error(self) -> None:
        with self._client(self._server()) as client:
            assert client.post("/seed_session", json={}).status_code == 200
            resp = client.post("/made_up_tool", json={"some_arg": "x"})
            assert resp.status_code == 200
            # Byte-equal to the historical dispatcher's KeyError soft error.
            assert resp.content == b"{\"output\":\"Error executing tool 'made_up_tool': 'made_up_tool'\"}"

    def test_none_arguments_are_filtered_before_dispatch(self) -> None:
        with self._client(self._server()) as client:
            assert client.post("/seed_session", json={}).status_code == 200
            resp = client.post("/company_directory_find_email_address", json={"name": None})
            assert resp.status_code == 200
            assert resp.json() == {"output": "Name not provided."}


class TestDualTransport:
    """The @gym_tool migration must keep the HTTP wire contract and add the MCP surface."""

    def _server(self) -> WorkbenchResourcesServer:
        config = WorkbenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="workplace_assistant")
        return WorkbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _mock_analytics_reset_state(self, analytics_tool_instance):
        analytics_tool_instance._analytics_data = pd.DataFrame([{"user_engaged": "False"}])
        analytics_tool_instance._plots_data = pd.DataFrame(columns=["file_path"])

    def test_mcp_round_trip(self) -> None:
        mock_data = {"email_address": ["aisha.chen@atlas.com", "carlos.rodriguez@atlas.com"]}
        mock_df = pd.DataFrame(mock_data)
        with (
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.company_directory.pd.read_csv"
            ) as mock_read_csv,
            patch(
                "resources_servers.workplace_assistant.workplace_assistant_tools.analytics.AnalyticsTool.reset_state",
                side_effect=self._mock_analytics_reset_state,
                autospec=True,
            ) as _,
        ):
            mock_read_csv.return_value = mock_df
            server = self._server()

            with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
                seed_body = client.post("/seed_session", json={}).json()
                token = seed_body["mcp"]["headers"][TOKEN_HEADER]
                mcp_handshake(client, token=token)

                tools = mcp_list_tools(client, token=token)
                assert set(tools) == EXPECTED_TOOLS
                # The hand-authored schema dict is advertised verbatim over MCP.
                schemas = {schema["name"]: schema for schema in get_tools(TOOLKITS)["schemas"]}
                assert (
                    tools["company_directory_find_email_address"]["inputSchema"]
                    == schemas["company_directory_find_email_address"]["parameters"]
                )

                result = mcp_call(client, "company_directory_find_email_address", {"name": "aisha"}, token=token)
                assert result.get("isError") is not True
                assert result["structuredContent"] == {"output": ["aisha.chen@atlas.com"]}

    def test_mcp_call_without_token_is_tool_error(self) -> None:
        server = self._server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            result = mcp_call(client, "company_directory_find_email_address", {"name": "aisha"}, token=None)
            assert result["isError"] is True
            assert TOKEN_HEADER in result["content"][0]["text"]

    def test_transport_parity(self) -> None:
        server = self._server()
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            assert_transport_parity(app, client, EXPECTED_TOOLS)
        assert http_tool_names(app) == set(get_tools(TOOLKITS)["functions"])


class TestMCPNamespacedScoring:
    """MCP-driven rollouts (mcp__<server>__<tool> names) must score identically to HTTP ones."""

    def test_namespaced_and_bare_trajectories_score_the_same(self) -> None:
        import json

        dataset = json.loads(open("resources_servers/workplace_assistant/data/example.jsonl").readline())
        gt_calls = (
            eval(dataset["ground_truth"]) if isinstance(dataset["ground_truth"], str) else dataset["ground_truth"]
        )

        def response_with(names):
            return {
                "id": "r",
                "created_at": 0,
                "model": "t",
                "object": "response",
                "output": [
                    {
                        "type": "function_call",
                        "name": n,
                        "arguments": c["arguments"],
                        "call_id": f"c{i}",
                        "id": f"fc{i}",
                        "status": "completed",
                    }
                    for i, (n, c) in enumerate(zip(names, gt_calls))
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }

        from fastapi.testclient import TestClient

        config = WorkbenchResourcesServerConfig(host="", port=0, entrypoint="", name="workplace_assistant")
        server = WorkbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            bare = [c["name"] for c in gt_calls]
            namespaced = [f"mcp__workplace_assistant__{n}" for n in bare]
            rewards = []
            for names in (bare, namespaced):
                vr = client.post(
                    "/verify",
                    json={
                        "id": dataset["id"],
                        "responses_create_params": dataset["responses_create_params"],
                        "ground_truth": dataset["ground_truth"],
                        "category": dataset["category"],
                        "environment_name": dataset["environment_name"],
                        "response": response_with(names),
                    },
                )
                assert vr.status_code == 200, vr.text
                rewards.append(vr.json()["reward"])
            assert rewards[0] == rewards[1] == 1.0, rewards
