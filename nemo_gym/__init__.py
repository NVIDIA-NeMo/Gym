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
from os.path import abspath, dirname, join


ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")
RESULTS_DIR = join(PARENT_DIR, "results")

sys.path.append(PARENT_DIR)

# TODO: Maybe eventually we want an override for OMP_NUM_THREADS ?

# Turn off HF tokenizers paralellism
environ["TOKENIZERS_PARALLELISM"] = "false"

# Huggingface related caching directory overrides to local folders.
environ["HF_DATASETS_CACHE"] = join(CACHE_DIR, "huggingface")
environ["TRANSFORMERS_CACHE"] = environ["HF_DATASETS_CACHE"]
# TODO When `TRANSFORMERS_CACHE` is no longer supported in transformers>=5.0.0, migrate to `HF_HOME`
# environ["HF_HOME"] = join(CACHE_DIR, "huggingface")

# UV caching directory overrides to local folders.
environ["UV_CACHE_DIR"] = join(CACHE_DIR, "uv")

# Turn off Gradio analytics
environ["GRADIO_ANALYTICS_ENABLED"] = "False"

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
