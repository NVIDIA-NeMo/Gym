# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open the browser client for a running Kuhn Poker resources server."""

import argparse
import sys
import webbrowser

from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.server_utils import ServerClient


def play_url(server_name: str = "kuhn_poker") -> str:
    server_client = ServerClient.load_from_global_config()
    server_config = get_first_server_config_dict(server_client.global_config_dict, server_name)
    return f"http://{server_config.host}:{server_config.port}/play"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="kuhn_poker")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        url = play_url(args.server_name)
    finally:
        sys.argv = original_argv

    print(f"Kuhn Poker: {url}")
    if not args.no_open and not webbrowser.open(url):
        print("The browser could not be opened automatically. Open the URL above manually.")


if __name__ == "__main__":
    main()
