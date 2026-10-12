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

"""Capture Pi's JSON events at receipt time using only sandbox Python's stdlib."""

import json
import subprocess
import sys
from time import time


def main() -> None:
    with open(sys.argv[1], "w", buffering=1) as events:
        process = subprocess.Popen(sys.argv[2:], stdout=subprocess.PIPE)
        assert process.stdout is not None
        for line in process.stdout:
            observed_at = time()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            try:
                event = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if isinstance(event, dict):
                events.write(json.dumps([observed_at, event]) + "\n")
        raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
