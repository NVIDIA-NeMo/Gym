# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from resources_servers.gdpval.task_data import TaskData


@pytest.mark.parametrize("path", ["../x", "reference_files/../../x", "a/../b", "/abs/../x"])
def test_reference_paths_cannot_leave_their_target_directory(path: str) -> None:
    with pytest.raises(ValidationError, match="leaves its target directory"):
        TaskData(task_id="t", reference_files=[path])


@pytest.mark.parametrize("path", ["reference_files/abc/brief.txt", "/abs/brief.txt", "brief..txt"])
def test_reference_paths_inside_their_target_directory_are_accepted(path: str) -> None:
    assert TaskData(task_id="t", reference_files=[path]).reference_files == [path]
