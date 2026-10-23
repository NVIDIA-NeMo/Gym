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
"""Compatibility imports for rollout identity helpers, including NeMo-RL callers."""

from nemo_gym.rollout_sampling import (
    _ROLLOUT_ID as _ROLLOUT_ID,
)
from nemo_gym.rollout_sampling import (
    ROLLOUT_ID_PATTERN as ROLLOUT_ID_PATTERN,
)
from nemo_gym.rollout_sampling import (
    ROUTED_PATH_STATE_KEY as ROUTED_PATH_STATE_KEY,
)
from nemo_gym.rollout_sampling import (
    RolloutContextMiddleware as RolloutContextMiddleware,
)
from nemo_gym.rollout_sampling import (
    current_episode_id as current_episode_id,
)
from nemo_gym.rollout_sampling import (
    current_rollout_id as current_rollout_id,
)
from nemo_gym.rollout_sampling import (
    maybe_rollout_id_from_run_body as maybe_rollout_id_from_run_body,
)
from nemo_gym.rollout_sampling import (
    rollout_context as rollout_context,
)
from nemo_gym.rollout_sampling import (
    scope_with_routed_path as scope_with_routed_path,
)
