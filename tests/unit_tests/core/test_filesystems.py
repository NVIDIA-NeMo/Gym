# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from pathlib import Path

import pytest

from nemo_gym.core.filesystems import filesystem_type, warn_if_unsuitable


MOUNTS = """\
22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw
40 22 0:50 / /lustre rw,relatime shared:20 - lustre 10.0.0.1@o2ib:/fs rw
41 22 259:3 / /raid/scratch rw,relatime shared:21 - xfs /dev/nvme1n1 rw
42 40 0:51 / /lustre/local\\040copy rw,relatime shared:22 - ext4 /dev/sdb rw
43 22 0:52 / /tmp rw,nosuid shared:23 - tmpfs tmpfs rw
"""


@pytest.fixture
def mountinfo(tmp_path: Path) -> Path:
    path = tmp_path / "mountinfo"
    path.write_text(MOUNTS)
    return path


def test_the_deepest_mount_above_the_path_decides(mountinfo: Path) -> None:
    assert filesystem_type("/lustre/project/temp", mountinfo=mountinfo) == "lustre"
    assert filesystem_type("/raid/scratch/gym", mountinfo=mountinfo) == "xfs"
    assert filesystem_type("/lustre/local copy/x", mountinfo=mountinfo) == "ext4"
    assert filesystem_type("/home/user", mountinfo=mountinfo) == "ext4"
    # A sibling whose name only starts with the mount point is not under it.
    assert filesystem_type("/lustre2/x", mountinfo=mountinfo) == "ext4"


def test_no_mount_table_means_unknown(tmp_path: Path) -> None:
    assert filesystem_type("/lustre/x", mountinfo=tmp_path / "missing") is None


def test_warns_for_shared_and_memory_filesystems_only(mountinfo: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert warn_if_unsuitable("temp_dir", "/lustre/project/temp", mountinfo=mountinfo) == "lustre"
        assert warn_if_unsuitable("temp_dir", "/tmp/gym", mountinfo=mountinfo) == "tmpfs"
        assert warn_if_unsuitable("temp_dir", "/raid/scratch/gym", mountinfo=mountinfo) is None
        assert warn_if_unsuitable("temp_dir", None, mountinfo=mountinfo) is None

    assert len(caplog.records) == 2
    assert "temp_dir is /lustre/project/temp, on a lustre filesystem shared by many hosts" in caplog.text
    assert "temp_dir is /tmp/gym, on a tmpfs filesystem held in memory" in caplog.text


def test_the_caller_supplies_the_advice(mountinfo: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        warn_if_unsuitable(
            "model_call_capture_dir", "/lustre/run/model-calls", advice="Kept as run output.", mountinfo=mountinfo
        )

    assert caplog.text.rstrip().endswith("Kept as run output.")


def test_the_deepest_mount_wins_whatever_the_line_order(tmp_path: Path) -> None:
    mountinfo = tmp_path / "mountinfo"
    # The kernel lists mounts in mount order, so a parent mounted later can follow its child.
    mountinfo.write_text(
        "42 40 0:51 / /lustre/local rw,relatime shared:22 - ext4 /dev/sdb rw\n"
        "40 22 0:50 / /lustre rw,relatime shared:20 - lustre 10.0.0.1@o2ib:/fs rw\n"
        "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
    )

    assert filesystem_type("/lustre/local/x", mountinfo=mountinfo) == "ext4"
    assert filesystem_type("/lustre/x", mountinfo=mountinfo) == "lustre"
