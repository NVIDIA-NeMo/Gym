# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`fake-model --port N`: a scripted OpenAI chat-completions endpoint for native oracle and control runs.

Point the native arm's `policy_model` at it with `--model-url http://127.0.0.1:N/v1`. Each reply is the `text` of the
first `--replies` row whose `match` occurs in the last user message (as sent, or with OpenCode's argv quoting undone),
else `--reply`, else `no-match`. The replies file is re-read on every request, so it can change between runs.
"""

import argparse
import json
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.harbor_conversion_checks.messages import InputError, iter_jsonl, message_text, unquote_argv


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, required=True, help="port on 127.0.0.1 (0 picks a free one)")
    parser.add_argument("--replies", type=Path, help='JSONL of {"match": ..., "text": ...}, re-read per request')
    parser.add_argument("--reply", help="the reply when no --replies row matches")


def read_replies(path: Path) -> list[tuple[str, str]]:
    replies = []
    for index, row in iter_jsonl(path):
        if not isinstance(row.get("match"), str) or not isinstance(row.get("text"), str):
            raise InputError(f"{path} line {index + 1}: expected string fields match and text")
        replies.append((row["match"], row["text"]))
    return replies


class FakeModelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, *, replies: Path | None, reply: str | None) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.replies, self.reply, self.last_model = replies, reply, "fake-model"

    def reply_for(self, body: dict) -> str:
        users = [
            message_text(m.get("content"))
            for m in body.get("messages") or []
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        last = users[-1] if users else ""
        forms = [last] + [text for text in [unquote_argv(last)] if text is not None]
        for match, text in read_replies(self.replies) if self.replies else []:
            if any(match in form for form in forms):
                return text
        return self.reply if self.reply is not None else "no-match"


class _Handler(BaseHTTPRequestHandler):
    server: FakeModelServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") != "/v1/models":
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        self._send(200, {"object": "list", "data": [{"id": self.server.last_model, "object": "model"}]})

    def do_POST(self) -> None:
        if self.path.split("?")[0].rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if not isinstance(body, dict):
                raise json.JSONDecodeError("the request body is not a JSON object", "", 0)
            text = self.server.reply_for(body)
        except (json.JSONDecodeError, InputError) as exc:
            self._send(400 if isinstance(exc, json.JSONDecodeError) else 500, {"error": {"message": str(exc)}})
            return
        model = body.get("model") or self.server.last_model
        self.server.last_model = model
        usage = {"prompt_tokens": 1, "completion_tokens": max(1, len(text.split())), "total_tokens": 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        base = {"id": "chatcmpl-fake", "created": int(time.time()), "model": model}
        if not body.get("stream"):
            message = {"role": "assistant", "content": text}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            self._send(200, {**base, "object": "chat.completion", "choices": [choice], "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        final = {"index": 0, "delta": {}, "finish_reason": "stop"}
        chunks = [
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            },
            {**base, "choices": [final]},
        ]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks[-1]["usage"] = usage
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps({**chunk, "object": "chat.completion.chunk"}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _stop(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def run(args: argparse.Namespace) -> int:
    if args.replies is None and args.reply is None:
        raise InputError("pass --replies FILE, --reply TEXT or both")
    if args.replies is not None:
        read_replies(args.replies)
    try:
        server = FakeModelServer(args.port, replies=args.replies, reply=args.reply)
    except OSError as exc:
        raise InputError(f"cannot listen on 127.0.0.1:{args.port}: {exc}") from exc
    signal.signal(signal.SIGTERM, _stop)
    print(f"fake-model listening on http://127.0.0.1:{server.server_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
