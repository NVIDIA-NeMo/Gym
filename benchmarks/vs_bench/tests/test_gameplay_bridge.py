# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the real upstream episode loop with a fixture policy."""

import asyncio
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path


AUTHOR = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
AUTHOR_PYTHON = os.environ.get("VS_BENCH_TEST_AUTHOR_PYTHON", sys.executable)
WORKER = Path(__file__).resolve().parents[1] / "reference_gameplay_agent.py"


@unittest.skipUnless(AUTHOR.is_dir(), "Pinned author checkout required")
class GameplayBridgeTest(unittest.TestCase):
    def test_original_poker_suite_preserves_all_ten_source_episodes(self):
        async def exchange(episode_dir):
            process = await asyncio.create_subprocess_exec(
                AUTHOR_PYTHON,
                str(WORKER),
                str(AUTHOR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async def send(value):
                data = json.dumps(value).encode()
                process.stdin.write(struct.pack("!I", len(data)) + data)
                await process.stdin.drain()

            async def receive():
                size = struct.unpack("!I", await process.stdout.readexactly(4))[0]
                return json.loads(await process.stdout.readexactly(size))

            try:
                await send(
                    {
                        "mode": "suite",
                        "game": "kuhn_poker",
                        "episode_id": 0,
                        "episode_dir": str(episode_dir),
                        "reasoning_model": True,
                        "temperature": 1.0,
                        "max_response_tokens": 8192,
                        "max_reasoning_tokens": 32000,
                    }
                )
                while True:
                    frame = await receive()
                    if frame["type"] != "model_request":
                        break
                    await send(
                        {
                            "type": "model_response",
                            "request_id": frame["request_id"],
                            "content": '{"action":"<PASS>"}',
                            "finish_reason": "stop",
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                        }
                    )
                self.assertEqual(frame["type"], "complete", frame)
                self.assertEqual(len(frame["returns"]), 10)
                self.assertEqual(frame["model_seats"], [1])
                self.assertTrue(Path(frame["source_results_path"]).is_file())
                self.assertEqual(await process.wait(), 0)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(asyncio.wait_for(exchange(Path(tmp) / "suite"), timeout=90))

    def test_original_poker_episode_with_real_cfr_opponent(self):
        async def exchange(episode_dir):
            process = await asyncio.create_subprocess_exec(
                AUTHOR_PYTHON,
                str(WORKER),
                str(AUTHOR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def send(value):
                data = json.dumps(value).encode()
                process.stdin.write(struct.pack("!I", len(data)) + data)
                await process.stdin.drain()

            async def receive():
                size = struct.unpack("!I", await process.stdout.readexactly(4))[0]
                return json.loads(await process.stdout.readexactly(size))

            try:
                await send(
                    {
                        "game": "kuhn_poker",
                        "episode_id": 0,
                        "episode_dir": str(episode_dir),
                        "reasoning_model": True,
                        "temperature": 1.0,
                        "max_response_tokens": 8192,
                        "max_reasoning_tokens": 32000,
                    }
                )
                while True:
                    frame = await receive()
                    if frame["type"] != "model_request":
                        break
                    self.assertEqual(frame["request"]["max_completion_tokens"], 24192)
                    self.assertTrue(
                        any(
                            part["type"] == "image_url"
                            for msg in frame["request"]["messages"]
                            for part in msg["content"]
                        )
                    )
                    await send(
                        {
                            "type": "model_response",
                            "request_id": frame["request_id"],
                            "content": '{"action":"<PASS>"}',
                            "finish_reason": "stop",
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                        }
                    )
                self.assertEqual(frame["type"], "complete", frame)
                self.assertGreater(frame["steps"], 0)
                self.assertGreater(frame["model_calls"], 0)
                self.assertEqual(len(frame["returns"]), 2)
                self.assertEqual(await process.wait(), 0, (await process.stderr.read()).decode()[-1000:])
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(asyncio.wait_for(exchange(Path(tmp) / "episode"), timeout=90))
