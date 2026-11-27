# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the converter with actual rows and the original model adapter."""

import asyncio
import importlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from benchmarks.vs_bench import prepare


DATASET = Path(os.environ.get("VS_BENCH_TEST_DATASET", "/nonexistent"))
UPSTREAM = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))


@unittest.skipUnless(DATASET.is_dir() and UPSTREAM.is_dir(), "Pinned dataset and upstream reference required")
class PreparedRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the actual upstream adapter without its unrelated eager provider imports.
        cls.saved = {name: sys.modules.get(name) for name in ("models", "utils", "agents", "envs")}
        for name in cls.saved:
            package = types.ModuleType(name)
            package.__path__ = [str(UPSTREAM / name)]
            sys.modules[name] = package
        sys.path.insert(0, str(UPSTREAM))
        cls.local = importlib.import_module("models.local")
        cls.base = importlib.import_module("models.base_model")
        cls.agent_module = importlib.import_module("agents.prompt_agent")
        cls.env_base = importlib.import_module("envs.base_env")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(UPSTREAM))
        for name in list(sys.modules):
            if any(name == package or name.startswith(package + ".") for package in cls.saved):
                sys.modules.pop(name, None)
        for name, previous in cls.saved.items():
            if previous is not None:
                sys.modules[name] = previous

    def test_original_prompt_and_images_match_both_model_styles(self):
        for track, game in (("reasoning", "kuhn_poker"), ("perception", "coin_dilemma")):
            raw = next(pq.ParquetFile(DATASET / track / f"{game}.parquet").iter_batches(batch_size=1)).to_pylist()[0]
            original_prompt = self.base.Prompt(
                raw["prompt"]["system"],
                self.base.ObservationPrompt(raw["prompt"]["observation"], raw["prompt"]["image_path"]),
                raw["prompt"]["action"] + "\n\n" + prepare.FINAL_INSTRUCTION,
            )
            for reasoning_model in (True, False):
                adapter = object.__new__(self.local.LocalModel)
                adapter.reasoning_model = reasoning_model
                original = adapter._get_messages(original_prompt, log=False, reasoning=True)
                converted = prepare.convert_row(track, game, 0, raw, reasoning_model=reasoning_model)
                expected = []
                for message in original:
                    blocks = []
                    for block in message["content"]:
                        if block["type"] == "text":
                            blocks.append({"type": "input_text", "text": block["text"]})
                        else:
                            blocks.append(
                                {
                                    "type": "input_image",
                                    "image_url": block["image_url"]["url"],
                                    "detail": block["image_url"]["detail"],
                                }
                            )
                    expected.append({"role": message["role"], "content": blocks})
                self.assertEqual(converted["responses_create_params"]["input"], expected)
                if track == "reasoning":
                    self.assertEqual(converted["verifier_metadata"]["next_action"], raw["next_action"])
                    self.assertEqual(
                        converted["verifier_metadata"]["next_legal_actions"], json.loads(raw["next_legal_actions"])
                    )
                else:
                    self.assertEqual(converted["verifier_metadata"]["label"], raw["label"])

    def test_streamed_preparation_is_bounded_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "offline.jsonl"
            prepare.prepare(DATASET, output, tracks=("reasoning", "perception"), games=("kuhn_poker",), limit=2)
            original = output.read_bytes()
            result = [json.loads(line) for line in original.splitlines()]
            self.assertEqual(
                [(r["verifier_metadata"]["track"], r["verifier_metadata"]["row_index"]) for r in result],
                [("reasoning", 0), ("reasoning", 1), ("perception", 0), ("perception", 1)],
            )
            with self.assertRaises(FileExistsError):
                prepare.prepare(DATASET, output, tracks=("reasoning",), games=("kuhn_poker",), limit=1)
            self.assertEqual(output.read_bytes(), original)

    def test_reasoning_bridge_invokes_author_parser_on_all_games(self):
        worker = Path(prepare.__file__).with_name("reference_reasoning.py")
        for game in prepare.GAMES:
            row = next(
                pq.ParquetFile(DATASET / "reasoning" / f"{game}.parquet").iter_batches(batch_size=1)
            ).to_pylist()[0]
            legal = json.loads(row["next_legal_actions"])
            expected = int(row["next_action"][0])
            payload = {
                "game": game,
                "answer": json.dumps({"action": legal[str(expected)]}),
                "legal_actions": legal,
                "next_action": row["next_action"],
            }
            run = subprocess.run(
                [sys.executable, str(worker), str(UPSTREAM)],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(run.returncode, 0, f"{game}: {run.stderr[-1000:]}")
            result = json.loads(run.stdout)
            self.assertTrue(result["valid"], game)
            self.assertTrue(result["correct"], game)
        payload["answer"] = "No action was given."
        run = subprocess.run(
            [sys.executable, str(worker), str(UPSTREAM)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        self.assertEqual(json.loads(run.stdout), {"valid": False, "outcome": "retryable_format_error"})
        payload["answer"] = '{"action": "definitely-illegal"}'
        run = subprocess.run(
            [sys.executable, str(worker), str(UPSTREAM)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        self.assertEqual(
            json.loads(run.stdout),
            {"valid": False, "outcome": "illegal_action", "action": -99999999, "correct": False},
        )

    def test_author_predict_retries_bad_format_but_stops_for_illegal_action(self):
        raw = next(pq.ParquetFile(DATASET / "reasoning/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
        raw["next_legal_actions"] = json.loads(raw["next_legal_actions"])
        env_module = importlib.import_module("envs.kuhn_poker")
        patterns = object.__new__(env_module.KuhnPoker).regex_patterns
        good = json.dumps({"action": raw["next_legal_actions"][str(raw["next_action"][0])]})

        class Model:
            name = "reference-test"

            def __init__(self, answers):
                self.answers = list(answers)
                self.calls = 0

            async def generate(self, prompt, reasoning=False):
                self.calls += 1
                return [], None, self.answers.pop(0), {}

        async def no_wait(_):
            return None

        agent = object.__new__(self.agent_module.PromptAgent)
        agent.max_errors = 2
        agent.model = Model(["No action.", good])
        with patch.object(self.agent_module.asyncio, "sleep", no_wait):
            action, _ = asyncio.run(agent.predict(raw, patterns))
        self.assertIn(action, raw["next_action"])
        self.assertEqual(agent.model.calls, 2)

        agent.model = Model(['{"action": "definitely-illegal"}', good])
        with patch.object(self.agent_module.asyncio, "sleep", no_wait):
            action, _ = asyncio.run(agent.predict(raw, patterns))
        self.assertEqual(action, -99999999)
        self.assertEqual(agent.model.calls, 1)

    def test_framed_bridge_runs_original_agent_and_retries(self):
        raw = next(pq.ParquetFile(DATASET / "reasoning/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
        raw["next_legal_actions"] = json.loads(raw["next_legal_actions"])
        good = json.dumps({"action": raw["next_legal_actions"][str(raw["next_action"][0])]})
        worker = Path(prepare.__file__).with_name("reference_reasoning_agent.py")
        task = {
            "game": "kuhn_poker",
            "reasoning_model": True,
            "temperature": 1.0,
            "max_response_tokens": 8192,
            "max_reasoning_tokens": 32000,
            "max_errors": 20,
            "dataset_row": raw,
        }

        async def exchange():
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(worker),
                str(UPSTREAM),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async def send_frame(value):
                data = json.dumps(value).encode()
                process.stdin.write(struct.pack("!I", len(data)) + data)
                await process.stdin.drain()

            async def read_frame():
                length = struct.unpack("!I", await process.stdout.readexactly(4))[0]
                return json.loads(await process.stdout.readexactly(length))

            try:
                await send_frame(task)
                first = await read_frame()
                self.assertEqual(first["type"], "model_request")
                self.assertEqual(first["request"]["max_completion_tokens"], 24192)
                self.assertEqual(first["request"]["temperature"], 1.0)
                for answer in ("No action.", good):
                    await send_frame(
                        {
                            "type": "model_response",
                            "content": answer,
                            "finish_reason": "stop",
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                        }
                    )
                    result = await read_frame()
                    if answer == "No action.":
                        self.assertEqual(result["type"], "model_request")
                self.assertEqual(result["type"], "complete")
                self.assertTrue(result["correct"])
                self.assertEqual(result["model_calls"], 2)
                self.assertEqual(result["exception_types"], ["RegexError"])
                self.assertEqual(await process.wait(), 0)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        asyncio.run(asyncio.wait_for(exchange(), timeout=45))

    def test_perception_bridge_calls_original_agent(self):
        raw = next(pq.ParquetFile(DATASET / "perception/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
        worker = Path(prepare.__file__).with_name("reference_reasoning_agent.py")
        task = {
            "track": "perception",
            "game": "kuhn_poker",
            "reasoning_model": True,
            "temperature": 1.0,
            "max_response_tokens": 8192,
            "max_reasoning_tokens": 32000,
            "max_errors": 20,
            "dataset_row": {"prompt": raw["prompt"]},
        }

        async def exchange():
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(worker),
                str(UPSTREAM),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async def send(value):
                data = json.dumps(value).encode()
                process.stdin.write(struct.pack("!I", len(data)) + data)
                await process.stdin.drain()

            async def receive():
                length = struct.unpack("!I", await process.stdout.readexactly(4))[0]
                return json.loads(await process.stdout.readexactly(length))

            try:
                await send(task)
                request = await receive()
                self.assertEqual(request["type"], "model_request")
                self.assertEqual(request["request"]["max_completion_tokens"], 24192)
                await send(
                    {
                        "type": "model_response",
                        "content": "{}",
                        "finish_reason": "stop",
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                    }
                )
                result = await receive()
                self.assertEqual(result["answer"], "{}")
                self.assertEqual(result["model_calls"], 1)
                self.assertEqual(await process.wait(), 0)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        asyncio.run(asyncio.wait_for(exchange(), timeout=45))

    def test_reference_perception_scores_actual_pandas_label(self):
        import pandas as pd

        label = pd.read_parquet(DATASET / "perception/kuhn_poker.parquet").to_dict(orient="index")["0"]["label"]
        key = next(iter(label))
        worker = Path(prepare.__file__).with_name("reference_perception_score.py")
        request = {"game": "kuhn_poker", "row_index": 0, "answer": repr({key: label[key]})}
        result = subprocess.run(
            [sys.executable, str(worker), str(UPSTREAM), str(DATASET)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), {"reward": 1.0})

    def test_fixed_kaz_scorer_handles_released_numpy_labels(self):
        import pandas as pd

        labels = pd.read_parquet(DATASET / "perception/knights_archers_zombies.parquet", columns=["label"])
        index = next(i for i in range(300) if len(labels.loc[str(i), "label"]["Zombies"]) > 1)
        worker = Path(prepare.__file__).with_name("reference_perception_score.py")
        result = subprocess.run(
            [sys.executable, str(worker), str(UPSTREAM), str(DATASET)],
            input=json.dumps(
                {"game": "knights_archers_zombies", "row_index": index, "answer": '{"Zombies": [[0, 0]]}'}
            ),
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        reward = json.loads(result.stdout)["reward"]
        self.assertTrue(0.0 <= reward <= 1.0)

        label = labels.loc[str(index), "label"]
        answer = {
            "Archer": label["Archer"][0].tolist(),
            "Knight": label["Knight"][0].tolist(),
            "Zombies_count": int(label["Zombies_count"]),
            "Zombies": [z.tolist() for z in label["Zombies"]],
        }
        result = subprocess.run(
            [sys.executable, str(worker), str(UPSTREAM), str(DATASET)],
            input=json.dumps({"game": "knights_archers_zombies", "row_index": index, "answer": json.dumps(answer)}),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        self.assertAlmostEqual(json.loads(result.stdout)["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
