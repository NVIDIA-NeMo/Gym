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

"""Write the GDPVal dataset where the evaluation pipeline expects an agent's data to live.

The benchmark already knows how to build the dataset; this only re-homes it and applies
the limit, so the pipeline's per-agent prepare convention works unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from benchmarks.gdpval.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None, help="Accepted for interface parity; unused.")
    args = parser.parse_args()

    source = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.limit is None:
        shutil.copyfile(source, args.output)
    else:
        with source.open() as src, args.output.open("w") as dst:
            for row_count, line in enumerate(src):
                if row_count >= args.limit:
                    break
                dst.write(line)

    print(f"Wrote {sum(1 for _ in args.output.open())} tasks to {args.output}")


if __name__ == "__main__":
    main()
