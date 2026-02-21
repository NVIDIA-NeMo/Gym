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
import sys
from os import environ
from pathlib import Path


# /path/to/dir/Gym (PARENT_DIR)
# |- cache (CACHE_DIR)
# |- results (RESULTS_DIR)
# |- nemo_gym (ROOT_DIR)
# |- responses_api_models
# |- responses_api_agents
# ...
ROOT_DIR = Path(__file__).absolute().parent
PARENT_DIR = ROOT_DIR.parent
CACHE_DIR = PARENT_DIR / "cache"
RESULTS_DIR = PARENT_DIR / "results"

sys.path.append(str(PARENT_DIR))

# TODO: Maybe eventually we want an override for OMP_NUM_THREADS ?

# Turn off HF tokenizers paralellism
environ["TOKENIZERS_PARALLELISM"] = "false"

# Huggingface related caching directory overrides to local folders.
environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface")
environ["TRANSFORMERS_CACHE"] = environ["HF_DATASETS_CACHE"]
# TODO When `TRANSFORMERS_CACHE` is no longer supported in transformers>=5.0.0, migrate to `HF_HOME`
# environ["HF_HOME"] = join(CACHE_DIR, "huggingface")

from nemo_gym.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
