# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open the browser client for a running Kuhn Poker resources server."""

import argparse
import webbrowser

from nemo_gym.server_utils import get_server_url


def play_url(server_name: str = "kuhn_poker") -> str:
    return f"{get_server_url(server_name).rstrip('/')}/play"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="kuhn_poker")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()

    url = play_url(args.server_name)
    print(f"Kuhn Poker: {url}")
    if not args.no_open and not webbrowser.open(url):
        print("The browser could not be opened automatically. Open the URL above manually.")


if __name__ == "__main__":
    main()
