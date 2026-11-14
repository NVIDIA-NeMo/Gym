# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import importlib

import requests
from pytest import MonkeyPatch, fixture

from resources_servers.newton_bench.setup_newton_bench import ensure_newton_bench


def pytest_configure(config):
    ensure_newton_bench()


@fixture(autouse=True)
def placeholder_judge_key(monkeypatch: MonkeyPatch) -> None:
    """Give NewtonBench's symbolic judge a placeholder key, so setup_webserver starts, and keep it off the network.

    A test that reaches the judge sees each API attempt fail at once, unless it patches ``requests.post`` itself.
    """
    keys = importlib.import_module("utils.call_llm_api").keys
    monkeypatch.setitem(keys, "oa", None)
    monkeypatch.setitem(keys, "or", "placeholder-key")

    def no_network(*args, **kwargs):
        raise requests.ConnectionError("newton_bench tests do not reach the judge API")

    monkeypatch.setattr("requests.post", no_network)
