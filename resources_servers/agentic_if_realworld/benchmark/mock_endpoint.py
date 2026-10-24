# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local synthetic-only policy/judge HTTP double for plumbing smoke checks."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .synthetic import EXAMPLE_LITERALS, review


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "synthetic-judge":
            payload = json.loads(body["messages"][0]["content"].split("UNTRUSTED_EVALUATION_PAYLOAD:\n", 1)[1])
            answer = review(payload)
            value = {
                "id": "synthetic-judge-response",
                "created": 0,
                "model": "synthetic-judge",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(answer)},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())
        elif body["model"] == "synthetic-policy" and any(
            "Return exactly " + literal in body["messages"][0]["content"] for literal in EXAMPLE_LITERALS.values()
        ):
            literal = next(
                literal
                for literal in EXAMPLE_LITERALS.values()
                if "Return exactly " + literal in body["messages"][0]["content"]
            )
            # The five-example fixture exercises three passes and two complete failing answers.
            answer = {"YES.": "yes.", "NO.": "NO!"}.get(literal, literal)
            if not body.get("stream"):
                value = {
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "synthetic-policy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())
                return
            # Deliberately small independent synthetic stream; no real models/cases accepted.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            value = {
                "id": "synthetic",
                "created": 0,
                "model": "synthetic-policy",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": "stop"}],
            }
            self.wfile.write(("data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n").encode())
        else:
            self.send_error(400, "Synthetic fixture models only")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=19291)
    a = p.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
