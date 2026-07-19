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
from pathlib import Path
from typing import Optional

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ConfigPathNotFoundError


def resolve_input_path(input_path: str | Path, error_msg: Optional[str] = None) -> Path:
    _input_path = Path(input_path)
    if not _input_path.is_absolute():
        _cwd_path = Path.cwd() / input_path
        _input_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / input_path
    if not _input_path.is_file():
        error_msg = (
            error_msg or f"Given input file not found: '{input_path}'. Check it is spelled correctly and exists."
        )
        raise ConfigPathNotFoundError(error_msg)
    return _input_path


def failures_path_for(output_fpath: Path) -> Path:
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")
