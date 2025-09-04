# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock
from pytest import fixture
import pandas as pd
from nemo_gym.server_utils import ServerClient

from resources_servers.workbench.app import (
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    WorkbenchRequest,
    WorkbenchVerifyRequest,
)

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)

from unittest.mock import patch


class TestApp:
    @fixture
    def config(self) -> WorkbenchResourcesServerConfig:
        return WorkbenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )

    def init_server(self, config: WorkbenchResourcesServerConfig):
        server_mock = MagicMock(spec=ServerClient)
        resources_server = WorkbenchResourcesServer(
            config=config, server_client=server_mock
        )
        return resources_server

    async def test_company_directory_find_email_address(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = {
            "email_address": ["aisha.chen@atlas.com", "carlos.rodriguez@atlas.com"]
        }
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.company_directory.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(**{"name": "aisha"})

            response = await resources_server.route_to_python_function(
                path="company_directory_find_email_address", body=mock_request
            )

            assert response.output == ["aisha.chen@atlas.com"]

    async def test_email_get_email_information_by_id(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = {
            "email_id": ["00000393", "00000123"],
            "inbox/outbox": ["inbox", "outbox"],
            "sender/recipient": ["raj.patel@atlas.com", "test@example.com"],
            "subject": ["Update on Supply Chain", "Test Subject"],
            "sent_datetime": ["2023-10-01 13:11:52", "2023-10-01 14:00:00"],
            "body": ["Dear Sam, I have some ideas...", "This is a test body."],
        }
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.email.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(
                **{"email_id": "00000393", "field": "inbox/outbox"}
            )

            response = await resources_server.route_to_python_function(
                path="email_get_email_information_by_id", body=mock_request
            )

            assert response.output == {"inbox/outbox": "inbox"}

    async def test_email_search_emails(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "email_id": "match_01",
                "inbox/outbox": "inbox",
                "sender/recipient": "aisha.chen@atlas.com",  # <- This matches the query
                "subject": "Matching Email Subject",
                "sent_datetime": "2025-09-01 10:00:00",
                "body": "This is a test email from Aisha.",
            },
            {
                "email_id": "no_match_01",
                "inbox/outbox": "inbox",
                "sender/recipient": "another.user@example.com",  # <- This does NOT match
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

        with patch(
            "resources_servers.workbench.workbench_tools.email.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(**{"query": "aisha.chen@atlas.com"})

            response = await resources_server.route_to_python_function(
                path="email_search_emails", body=mock_request
            )

            assert response.output == expected_output

    async def test_email_send_email(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.email.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(
                **{
                    "recipient": "aisha.chen@atlas.com",
                    "subject": "regarding something",
                    "body": "some body",
                }
            )

            response = await resources_server.route_to_python_function(
                path="email_send_email", body=mock_request
            )
            assert response.output == "Email sent successfully."

    async def test_email_delete_email(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "email_id": "00000393",  # The target for deletion
                "inbox/outbox": "inbox",
                "sender/recipient": "user.one@example.com",
                "subject": "Email to be deleted",
                "sent_datetime": "2025-01-01 12:00:00",
                "body": "This is a test email.",
            },
            {
                "email_id": "00000123",  # Another email that should not be deleted
                "inbox/outbox": "inbox",
                "sender/recipient": "user.two@example.com",
                "subject": "Another Email",
                "sent_datetime": "2025-01-01 12:05:00",
                "body": "This is another test email.",
            },
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.email.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(**{"email_id": "00000393"})

            response = await resources_server.route_to_python_function(
                path="email_delete_email", body=mock_request
            )

            assert response.output == "Email deleted successfully."

    async def test_email_forward_email(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "email_id": "00000393",  # The target email to forward
                "inbox/outbox": "inbox",
                "sender/recipient": "original.sender@example.com",
                "subject": "Original Subject",
                "sent_datetime": "2025-01-01 12:00:00",
                "body": "This is the original email body.",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.email.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(
                **{"email_id": "00000393", "recipient": "aisha.chen@atlas.com"}
            )

            response = await resources_server.route_to_python_function(
                path="email_forward_email", body=mock_request
            )
            assert response.output == "Email forwarded successfully."

    async def test_calendar_get_event_information_by_id(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "event_id": "00000013",  # The target event ID
                "event_name": "Test Event",
                "participant_email": "test.user@example.com",
                "event_start": "2025-09-02 10:00:00",
                "duration": "60",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.calendar.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(
                **{"event_id": "00000013", "field": "event_id"}
            )

            response = await resources_server.route_to_python_function(
                path="calendar_get_event_information_by_id", body=mock_request
            )

            assert response.output == {"event_id": "00000013"}

    async def test_calendar_search_events(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.calendar.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(**{"query": "sync up"})

            response = await resources_server.route_to_python_function(
                path="calendar_search_events", body=mock_request
            )

            assert response.output == expected_output

    async def test_calendar_create_event(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"event_id": "00000299", "event_name": "Existing Event"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.calendar.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "event_name": "some event name",
                    "participant_email": "some.email@example.com",
                    "event_start": "2023-08-01 09:00:00",
                    "duration": "90",
                }
            )

            response = await resources_server.route_to_python_function(
                path="calendar_create_event", body=mock_request
            )

            assert response.output == "00000300"

    async def test_calendar_delete_event(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {"event_id": "00000013", "event_name": "Event to Delete"},
            {"event_id": "00000014", "event_name": "Another Event"},
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.calendar.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(**{"event_id": "00000013"})

            response = await resources_server.route_to_python_function(
                path="calendar_delete_event", body=mock_request
            )

            assert response.output == "Event deleted successfully."

    async def test_calendar_update_event(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "event_id": "00000013",
                "event_name": "Initial Name",
                "participant_email": "test@example.com",
                "event_start": "2025-01-01 10:00:00",
                "duration": "60",  # The initial value
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.calendar.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{"event_id": "00000013", "field": "duration", "new_value": "100"}
            )

            response = await resources_server.route_to_python_function(
                path="calendar_update_event", body=mock_request
            )

            assert response.output == "Event updated successfully."

    async def test_analytics_engaged_users_count(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {"date_of_visit": "2023-10-22", "user_engaged": "True"},
            {"date_of_visit": "2023-10-22", "user_engaged": "True"},
            {"date_of_visit": "2023-10-22", "user_engaged": "True"},
            {"date_of_visit": "2023-10-22", "user_engaged": "False"},
            {"date_of_visit": "2023-10-22", "user_engaged": "False"},
        ]

        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(
                **{"time_min": "2023-10-22", "time_max": "2023-10-22"}
            )

            response = await resources_server.route_to_python_function(
                path="analytics_engaged_users_count", body=mock_request
            )

            assert response.output == {"2023-10-22": 3}

    async def test_analytics_get_visitor_information_by_id(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "visitor_id": "0860",  # The target visitor
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

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)

            mock_request = WorkbenchRequest(**{"visitor_id": "0860"})

            response = await resources_server.route_to_python_function(
                path="analytics_get_visitor_information_by_id", body=mock_request
            )

            assert response.output == [
                {
                    "date_of_visit": "2023-10-22",
                    "visitor_id": "0860",
                    "page_views": "8",
                    "session_duration_seconds": "4",
                    "traffic_source": "referral",
                    "user_engaged": False,
                }
            ]

    async def test_analytics_create_plot(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "date_of_visit": "2023-10-22",
                "page_views": "10",
                "user_engaged": "False",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "time_min": "2023-10-22",
                    "time_max": "2023-10-22",
                    "value_to_plot": "total_visits",
                    "plot_type": "bar",
                }
            )

            response = await resources_server.route_to_python_function(
                path="analytics_create_plot", body=mock_request
            )

            assert response.output == "plots/2023-10-22_2023-10-22_total_visits_bar.png"

    async def test_analytics_traffic_source_count(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "time_min": "2023-10-22",
                    "time_max": "2023-10-22",
                    "traffic_source": "referral",
                }
            )

            response = await resources_server.route_to_python_function(
                path="analytics_traffic_source_count", body=mock_request
            )

            assert response.output == {"2023-10-22": 2}

    async def test_analytics_total_visits_count(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {"date_of_visit": "2023-10-22", "user_engaged": "False"} for _ in range(13)
        ] + [{"date_of_visit": "2023-10-23", "user_engaged": "False"} for _ in range(2)]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{"time_min": "2023-10-22", "time_max": "2023-10-22"}
            )

            response = await resources_server.route_to_python_function(
                path="analytics_total_visits_count", body=mock_request
            )

            assert response.output == {"2023-10-22": 13}

    async def test_analytics_get_average_session_duration(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.analytics.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{"time_min": "2023-10-22", "time_max": "2023-10-22"}
            )

            response = await resources_server.route_to_python_function(
                path="analytics_get_average_session_duration", body=mock_request
            )

            assert response.output == {"2023-10-22": 15.76923076923077}

    async def test_project_management_get_task_information_by_id(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"task_id": "00000149", "task_name": "Test Task"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.project_management.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{"task_id": "00000149", "field": "task_id"}
            )

            response = await resources_server.route_to_python_function(
                path="project_management_get_task_information_by_id", body=mock_request
            )

            assert response.output == {"task_id": "00000149"}

    async def test_project_management_search_tasks(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.project_management.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "task_name": "",
                    "assigned_to_email": "leila.azizi@atlas.com",
                    "list_name": "Backlog",
                    "due_date": "2023-11-27",
                    "board": "Front end",
                }
            )

            response = await resources_server.route_to_python_function(
                path="project_management_search_tasks", body=mock_request
            )

            assert response.output == expected_tasks

    async def test_project_management_create_task(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "task_id": "00000299",
                "assigned_to_email": "leila.azizi@atlas.com",
                "task_name": "Existing Task",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.project_management.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "task_name": "Add animation to carousel",
                    "assigned_to_email": "leila.azizi@atlas.com",
                    "list_name": "Backlog",
                    "due_date": "2023-11-27",
                    "board": "Front end",
                }
            )

            response = await resources_server.route_to_python_function(
                path="project_management_create_task", body=mock_request
            )

            assert response.output == "00000300"

    async def test_project_management_delete_task(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"task_id": "00000149", "task_name": "A task to delete"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.project_management.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(**{"task_id": "00000149"})

            response = await resources_server.route_to_python_function(
                path="project_management_delete_task", body=mock_request
            )

            assert response.output == "Task deleted successfully."

    async def test_project_management_update_task(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [
            {
                "task_id": "00000149",
                "task_name": "Some Task",
                "board": "Front end",
            }
        ]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.project_management.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df

            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{"task_id": "00000149", "field": "board", "new_value": "Design"}
            )

            response = await resources_server.route_to_python_function(
                path="project_management_update_task", body=mock_request
            )

            assert response.output == "Task updated successfully."

    async def test_customer_relationship_manager_search_customers(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
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

        with patch(
            "resources_servers.workbench.workbench_tools.customer_relationship_manager.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "customer_name": "Jaden White",
                    "customer_email": "jaden.white@protracefoods",
                    "product_interest": "Hardware",
                    "status": "Won",
                    "assigned_to_email": "raj.patel@atlas.com",
                    "last_contact_date_min": "2023-11-30 23:59:00",
                    "last_contact_date_max": "2023-11-30 23:59:00",
                    "follow_up_by_min": "2023-12-13 23:5",
                }
            )

            response = await resources_server.route_to_python_function(
                path="customer_relationship_manager_search_customers", body=mock_request
            )

            assert response.output == {
                "customers": [
                    {
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
                ],
                "pagination": {
                    "total_customers": 1,
                    "page": 1,
                    "page_size": 5,
                    "total_pages": 1,
                },
            }

    async def test_customer_relationship_manager_update_customer(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"customer_id": "00000189", "customer_name": "old name"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.customer_relationship_manager.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "customer_id": "00000189",
                    "field": "customer_name",
                    "new_value": "new customer",
                }
            )
            response = await resources_server.route_to_python_function(
                path="customer_relationship_manager_update_customer", body=mock_request
            )

            assert response.output == "Customer updated successfully."

    async def test_customer_relationship_manager_add_customer(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"customer_id": "00000199", "customer_name": "Max ID Customer"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.customer_relationship_manager.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(
                **{
                    "customer_name": "Some customer",
                    "product_interest": "Hardware",
                    "status": "Won",
                    "assigned_to_email": "raj.patel@atlas.com",
                    "last_contact_date": "2023-11-30 23:59:00",
                    "follow_up_by": "2023-12-13 23:59:00",
                    "notes": "some notes",
                }
            )
            response = await resources_server.route_to_python_function(
                path="customer_relationship_manager_add_customer", body=mock_request
            )

            assert response.output == "00000200"

    async def test_customer_relationship_manager_delete_customer(
        self, config: WorkbenchResourcesServerConfig
    ) -> None:
        mock_data = [{"customer_id": "00000189", "customer_name": "Customer to Delete"}]
        mock_df = pd.DataFrame(mock_data)

        with patch(
            "resources_servers.workbench.workbench_tools.customer_relationship_manager.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.return_value = mock_df
            resources_server = self.init_server(config)
            mock_request = WorkbenchRequest(**{"customer_id": "00000189"})
            response = await resources_server.route_to_python_function(
                path="customer_relationship_manager_delete_customer", body=mock_request
            )

            assert response.output == "Customer deleted successfully."

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

        def mock_analytics_reset_state(analytics_tool_instance):
            """
            This function will be used as the new reset_state.
            It sets the necessary attributes but keeps 'user_engaged' as a string.
            """
            analytics_tool_instance._analytics_data = pd.DataFrame(
                [{"user_engaged": "False"}]  # Keep as a string
            )
            analytics_tool_instance._plots_data = pd.DataFrame(columns=["file_path"])

        with (
            patch(
                "resources_servers.workbench.workbench_tools.email.pd.read_csv"
            ) as mock_email_csv,
            patch(
                "resources_servers.workbench.workbench_tools.analytics.AnalyticsTool.reset_state",
                side_effect=mock_analytics_reset_state,
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
                category="workbench_email",
                environment_name="workbench",
                id="0",
            )

            verification_response = await resources_server.verify(verify_request)

            assert verification_response.reward == 1.0
