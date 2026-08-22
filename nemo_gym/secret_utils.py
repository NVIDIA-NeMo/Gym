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
from omegaconf import DictConfig, ListConfig, open_dict


def recursively_hide_secrets(dict_config: DictConfig) -> None:
    """Mask every token/key leaf in place with '****' so a config can be printed or exported.

    Used by the config parser and the exporters.
    """

    def hide(node) -> None:
        for k, v in list(node.items()):
            if isinstance(v, (DictConfig, dict)):
                hide(v)
            elif isinstance(v, (ListConfig, list)):
                if "token" in k or "key" in k:
                    node[k] = ["****"] * len(v)
                else:
                    for inner_v in v:
                        if isinstance(inner_v, (DictConfig, dict)):
                            hide(inner_v)
            else:
                if "token" in k or "key" in k:
                    node[k] = "****"

    with open_dict(dict_config):
        hide(dict_config)
