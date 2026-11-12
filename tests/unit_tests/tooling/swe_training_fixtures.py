# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NEXT_ROW = {
    "instance_id": "example-task",
    "workdir": "/testbed",
    "image_ref": "example/image:task",
    "patch": "hidden-gold",
    "test_patch": "hidden-test",
    "expected_output_json": '{"test_one": "PASSED"}',
    "responses_create_params": {"input": [{"role": "user", "content": "Fix the widget.\r\nIt should return 2."}]},
    "source_server": "swe_next",
    "harbor_task_id": "prepared-task",
}
