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
import argparse
import asyncio
import logging

from aviary.dataset_server import TaskDatasetServer
from resources_servers.aviary.repl_app import BixBenchREPLDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--port",
        type=int,
        default=8042,
        help="",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="",
    )

    args = parser.parse_args()

    dataset = BixBenchREPLDataset(split=args.split)

    server = TaskDatasetServer(
        dataset=dataset,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )

    if args.use_async:
        asyncio.run(server.astart())
    else:
        server.start()


if __name__ == "__main__":
    main()
