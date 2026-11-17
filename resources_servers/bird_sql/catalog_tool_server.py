# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto Ontology's catalog tools for one BIRD episode, run inside the episode's catalog sandbox.

The ``bird_sql`` catalog Resources Server (``catalog_app.py``) uploads this file into the sandbox and runs it with
the sandbox image's Python, the one that has Auto Ontology (https://github.com/NVIDIA/auto-ontology) and its catalog
of the BIRD databases. It imports nothing from NeMo Gym.

    python catalog_tool_server.py serve SOCKET EPISODE_JSON [ENV_FILE]
    python catalog_tool_server.py wait SOCKET TIMEOUT_S
    python catalog_tool_server.py call SOCKET REQUEST_JSON

``serve`` holds the episode: one Auto Ontology ``ToolContext`` for the task's database, which keeps what earlier calls
found, and the episode's tool budget. EPISODE_JSON is a file with the task's ``database`` and ``question`` and the
budget, ``max_tool_calls``. ENV_FILE holds settings the image keeps beside Auto Ontology, such as its source
databases (``CONNECTION_STRINGS``). It answers one JSON request per connection on a Unix socket. ``call`` sends
one request, such as ``{"tool": "run_sql", "args": {"sql": "SELECT 1"}}``, and prints the reply,
``{"output": "...", "accepted": false}``. ``wait`` returns once ``serve`` answers.

The tools call Auto Ontology's own functions, as its ReAct loop does: ``retrieve`` searches the catalog, ``run_sql``
checks a query against the catalog and runs it read-only, and ``submit`` checks and runs the final query. A submission
that passes ends the episode: later calls are refused.
"""

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional


TOOLS = ("retrieve", "run_sql", "submit")
_MAX_REQUEST_BYTES = 1 << 20


def load_env_file(path: Path) -> None:
    """Set each ``KEY=VALUE`` line of ``path`` that the environment does not already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class Episode:
    """One task's tools on one database."""

    def __init__(self, functions: dict[str, Any], context: Any, *, max_tool_calls: int) -> None:
        self._functions = functions
        self._context = context
        self._max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.submitted = False

    def call(self, tool: str, args: Any) -> dict[str, Any]:
        """Run one tool call and return what the model sees, and whether it was an accepted submission."""
        if self.submitted:
            return {"output": "The answer is already submitted; the episode is over.", "accepted": False}
        if tool not in TOOLS:
            return {"output": f"Unknown tool {tool!r}. Use {', '.join(TOOLS)}.", "accepted": False}
        if not isinstance(args, dict):
            return {"output": f"Wrong arguments for {tool}: expected an object.", "accepted": False}
        self.tool_calls += 1
        if self.tool_calls > self._max_tool_calls and tool != "submit":
            return {
                "output": f"The tool budget of {self._max_tool_calls} calls is spent. Submit your answer now.",
                "accepted": False,
            }
        try:
            if tool == "retrieve":
                return {"output": self._functions["retrieve"](self._context, **args), "accepted": False}
            if tool == "run_sql":
                return {"output": self._functions["run_sql"](self._context, **args), "accepted": False}
            output, submission = self._functions["submit"](self._context, **args)
        except TypeError as error:
            return {"output": f"Wrong arguments for {tool}: {error}", "accepted": False}
        except Exception as error:  # noqa: BLE001 - a failing tool is an observation, as in Auto Ontology's loop
            return {"output": f"{tool} failed: {error}", "accepted": False}
        self.submitted = submission is not None
        return {"output": output, "accepted": self.submitted}


def auto_ontology_episode(database: str, question: str, *, max_tool_calls: int) -> Episode:
    """The episode's tools on ``database``, set up as Auto Ontology's ReAct step sets them up for a SQL answer."""
    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
    from auto_ontology.retrieval.text_to_sql.react.loop import submit
    from auto_ontology.retrieval.text_to_sql.react.retrieve import retrieve
    from auto_ontology.retrieval.text_to_sql.react.run_sql import run_sql
    from auto_ontology.utils import get_data_objects_retriever, get_semantic_objects_retriever

    # The connectors the catalog was ingested with (CONNECTION_STRINGS), so their names match the catalog's.
    connectors = {connector.database_name: connector for connector in get_connectors()}
    if database not in connectors:
        raise KeyError(f"No source database {database!r}; the catalog has {sorted(connectors)}")
    data_retriever = get_data_objects_retriever()
    context = ToolContext(
        retriever=get_semantic_objects_retriever() or data_retriever,
        data_retriever=data_retriever,
        connectors=[connectors[database]],
        database_name=database,
        information_answers=False,
        question=question,
    )
    functions = {"retrieve": retrieve, "run_sql": run_sql, "submit": submit}
    return Episode(functions, context, max_tool_calls=max_tool_calls)


def _read_line(connection: socket.socket) -> bytes:
    data = b""
    while not data.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > _MAX_REQUEST_BYTES:
            raise ValueError("request too large")
    return data


def serve(socket_path: str, episode: Episode) -> None:
    """Answer requests on ``socket_path`` until the sandbox stops."""
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen()
    while True:
        connection, _ = server.accept()
        with connection:
            try:
                request = json.loads(_read_line(connection))
                if request.get("tool") == "ping":
                    reply = {"output": "ready", "accepted": False}
                else:
                    reply = episode.call(str(request.get("tool")), request.get("args") or {})
            except Exception as error:  # noqa: BLE001 - one bad request must not stop the episode's tools
                reply = {"output": f"Bad tool request: {error}", "accepted": False}
            connection.sendall(json.dumps(reply, default=str).encode() + b"\n")


def request(socket_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with client:
        client.connect(socket_path)
        client.sendall(json.dumps(payload).encode() + b"\n")
        return json.loads(_read_line(client))


def wait(socket_path: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return request(socket_path, {"tool": "ping"})["output"] == "ready"
        except OSError:
            time.sleep(0.2)
    return False


def main(argv: list[str]) -> int:
    command, rest = (argv[0], argv[1:]) if argv else ("", [])
    if command == "serve" and len(rest) in (2, 3):
        socket_path, episode = rest[0], json.loads(Path(rest[1]).read_text())
        env_file: Optional[str] = rest[2] if len(rest) == 3 else None
        if env_file:
            load_env_file(Path(env_file))
        serve(
            socket_path,
            auto_ontology_episode(
                episode["database"], episode.get("question", ""), max_tool_calls=int(episode["max_tool_calls"])
            ),
        )
        return 0
    if command == "wait" and len(rest) == 2:
        return 0 if wait(rest[0], float(rest[1])) else 1
    if command == "call" and len(rest) == 2:
        print(json.dumps(request(rest[0], json.loads(rest[1]))))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
