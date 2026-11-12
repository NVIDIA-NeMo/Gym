# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import litellm
import tyro
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from bbh_mcp.jupyter import JupyterClient, JupyterServer
from bbh_mcp.tool_utils import RUBRIC_SCORE_PROMPT


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BBHTask:
    answer: bool
    hypothesis: str
    id: str
    input_data_path: str | Path
    max_points: int
    nb_primary_language: str
    protocol: str
    rubric: str
    capsule_dir: Path

    def __post_init__(self):
        self.input_data_path = Path(self.input_data_path)
        while self.input_data_path.suffix:
            self.input_data_path = self.input_data_path.with_suffix("")

        self.capsule_dir = Path(self.capsule_dir)

        assert self.capsule
        assert self.language

    @property
    def language(self):
        return self.nb_primary_language.lower()

    @property
    def kernel_name(self):
        kernel_name_map = {
            "python": "python3",
            "r": "ir",
        }

        return kernel_name_map.get(self.language)

    @property
    def capsule(self) -> Path:
        candidates = [
            self.capsule_dir / self.input_data_path,
            self.capsule_dir / f"CapsuleData-{self.id}",
            self.capsule_dir / f"capsule_{self.id}",
            self.capsule_dir,
        ]
        for d in candidates:
            if d.is_dir():
                return d

        raise FileNotFoundError(f"Input data directory does not exist: {candidates[0]}")


@dataclass(slots=True)
class MCPConfig:
    """CLI and Jupyter runtime configuration for a benchmark task."""

    ## Task options
    problem_jsonl: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """File containing benchmark task per row as a JSON."""

    task_id: Annotated[int, tyro.conf.arg(aliases=["--id"])]
    """0-indexed task ID corresponding to the row of problem_jsonl file"""

    capsule_dir: Annotated[str | Path | None, tyro.conf.arg(aliases=["-d"])] = None
    """Path to data capsules; defaults to `capsules` folder in the same directory."""

    log_dir: str | Path | None = None
    """Optional directory for notebook and scoring metrics logs."""

    rubric_model: Annotated[str | None, tyro.conf.arg(aliases=["-m"])] = os.getenv("RUBRIC_MODEL")
    """Name of rubric model, consumed by litellm."""

    rubric_model_api_base: Annotated[str | None, tyro.conf.arg(aliases=["-l"])] = os.getenv("RUBRIC_MODEL_API_BASE")
    """Rubric model API base."""

    rubric_model_api_key: Annotated[str | None, tyro.conf.arg(aliases=["-k"])] = os.getenv("RUBRIC_MODEL_API_KEY")
    """Rubric model API key."""

    rubric_attempts: int = int(os.getenv("RUBRIC_ATTEMPTS", "2"))
    """Rubric model call attempts; together with rubric_timeout they must fit the MCP timeout in opencode.json."""

    rubric_timeout: float = float(os.getenv("RUBRIC_TIMEOUT", "120"))
    """Seconds per rubric model call."""

    rubric_max_tokens: int = int(os.getenv("RUBRIC_MAX_TOKENS", "16384"))
    """Token limit of the rubric model's answer, reasoning included."""

    task: BBHTask = field(init=False)

    ## MCP options
    name: str = "bbh"
    host: Annotated[str, tyro.conf.arg(aliases=["-h"])] = "127.0.0.1"
    port: Annotated[int, tyro.conf.arg(aliases=["-p"])] = 8000
    id_token: str = os.getenv("BBH_MCP_TOKEN", "")

    ## Jupyter server/client options
    backend: Annotated[Literal["local", "docker"], tyro.conf.arg(aliases=["-b"])] = "docker"

    kernel_workdir: Annotated[str | Path, tyro.conf.arg(aliases=["-w"])] = "/app"
    """Path to working directory where any new kernel is launched."""

    container_image: Annotated[str | None, tyro.conf.arg(aliases=["-i"])] = None
    """Container image used for Jupyter kernel isolation."""

    cell_timeout: float = float(60 * 60)
    """Per cell run timeout."""

    def __post_init__(self) -> None:
        # Keep judge settings in config, not in the environment inherited by Jupyter.
        # This does not isolate the grader from same-identity process inspection.
        for name in ("RUBRIC_MODEL", "RUBRIC_MODEL_API_BASE", "RUBRIC_MODEL_API_KEY"):
            os.environ.pop(name, None)

        if not self.id_token:
            self.id_token = secrets.token_hex(32)

        self.problem_jsonl = Path(self.problem_jsonl).expanduser()
        self.capsule_dir = (
            Path(self.capsule_dir).expanduser()
            if self.capsule_dir is not None
            else self.problem_jsonl.parent / "capsules"
        )

        if not self.problem_jsonl.is_file():
            raise FileNotFoundError(f"Tasks file does not exist: {self.problem_jsonl}")

        if not self.capsule_dir.is_dir():
            raise FileNotFoundError(f"Capsule directory does not exist: {self.capsule_dir}")

        if self.task_id < 0:
            raise ValueError("task row number must be non-negative")

        if self.log_dir is not None:
            self.log_dir = Path(self.log_dir).expanduser()

        self.kernel_workdir = Path(self.kernel_workdir).expanduser()

        with self.problem_jsonl.open(encoding="utf-8") as file:
            for row_number, line in enumerate(line for line in file if line.strip()):
                if row_number != self.task_id:
                    continue

                task = json.loads(line)
                if not isinstance(task, dict):
                    raise TypeError(f"task row {self.task_id} is not a JSON object")

                self.task = BBHTask(**task, capsule_dir=self.capsule_dir)

                return

        raise ValueError(f"task row {self.task_id} was not found in {self.problem_jsonl}")


class MCPServer:
    POST_SUBMIT_TOOL_RESPONSE: str = "An answer has already been submitted. End all work immediately."

    def __init__(self, config: MCPConfig):
        self.config = config

        self.jupyter_server = JupyterServer(
            config.backend,
            container_image=config.container_image,
            container_mounts=[(config.task.capsule, config.kernel_workdir / "data", True)],
            kernel_workdir=config.kernel_workdir,
        )
        self.jupyter_client = JupyterClient(
            self.jupyter_server,
            config.task.kernel_name,
            cell_timeout=config.cell_timeout,
        )

        self.mcp = FastMCP(config.name, host=config.host, port=config.port)
        self.register_endpoints()
        self.register_tools()

        ## Bookkeeping
        self._ready = False
        self._submitted = False
        self._submission_lock = asyncio.Lock()

    def start(self):
        async def run():
            try:
                await self.jupyter_server.start()
                self._ready = True
                await self.mcp.run_streamable_http_async()
            finally:
                self._ready = False
                await self.jupyter_server.stop()

        asyncio.run(run())

    def register_endpoints(self):
        @self.mcp.custom_route("/health", methods=["GET"])
        async def health(_: Request) -> JSONResponse:
            if self._ready:
                return JSONResponse({"status": "ok", "id_token": self.config.id_token})
            return JSONResponse({"status": "starting"}, status_code=503)

    def register_tools(self):
        @self.mcp.tool()
        async def reset_kernel() -> str:
            """Reset the kernel to a clean state.

            This tool clears all variables and execution state.
            """
            if self._submitted:
                return self.POST_SUBMIT_TOOL_RESPONSE

            try:
                await self.jupyter_client.reset_kernel()
            except Exception:
                logger.exception("Kernel reset failed.")
                return "Internal server error while resetting kernel."

            return "Kernel reset successfully."

        @self.mcp.tool()
        async def run_cell(code: str, idx: int | None = None) -> str:
            """Run code in a notebook cell and return the execution output.

            This method allows running code in a new cell (append) or re-running
            an existing cell with updated code.

            Usage Examples:
                run_cell("print('Hello, world!')")           # Run code in new cell
                run_cell("print('Hello, world!')", idx=0)    # Run code in existing cell at index 0

            Error Recovery:
                When a cell fails with an error, you MUST fix it by calling run_cell
                with the corrected code and the SAME idx as the failed cell:

                run_cell("corrected_code", idx=3)  # Fix error in Cell #3

                The cell number is shown in the output prefix (e.g., "[Cell #3]").
                Do NOT create a new cell to fix an error - always edit the failed cell.

            Args:
                code: Code to execute
                idx: Cell index to run. If None or >= len(cells), appends a new cell.
                    If provided, updates and re-runs the existing cell at that index.
                    Use this to fix errors in existing cells.
            """
            if self._submitted:
                return self.POST_SUBMIT_TOOL_RESPONSE

            try:
                return await self.jupyter_client.run_cell(code, idx=idx)
            except Exception:
                logger.exception("Kernel run cell failed")
                return "Internal server error while running cell."

        @self.mcp.tool()
        async def list_dir(directory: str = ".", max_files: int = 20, show_hidden: bool = False) -> str:
            """List contents of a directory with truncation protection.

            Recursively lists files in a directory, with built-in protection against
            overwhelming the context with too many files. Use this tool instead of
            writing code to list directories to avoid context bloat.

            Usage Examples:
                list_dir()                      # List working directory
                list_dir("data/")               # List specific folder
                list_dir(max_files=50)          # Show more files
                list_dir(show_hidden=True)      # Include hidden files

            Args:
                directory: Directory path to list (default: current working directory)
                max_files: Maximum number of files to display (default: 20)
                show_hidden: Whether to show hidden files starting with '.' (default: False)
            """
            if self._submitted:
                return self.POST_SUBMIT_TOOL_RESPONSE

            try:
                return await self.jupyter_client.list_dir(directory, max_files, show_hidden)
            except Exception:
                logger.exception("List directory tool failed.")
                return "Error listing directory."

        @self.mcp.tool()
        async def submit_answer(answer: str) -> str:
            """Submit your response to the research question.

            Note that this tool may only be called once and ends the episode.

            Args:
                answer: Your final response to the research question
            """
            async with self._submission_lock:
                return await submit_once(answer)

        async def submit_once(answer: str) -> str:
            if self._submitted:
                return self.POST_SUBMIT_TOOL_RESPONSE

            notebook = self.jupyter_client.notebook.render(self.config.task.language)
            prompt = RUBRIC_SCORE_PROMPT.format(
                hypothesis=self.config.task.hypothesis,
                accepted=self.config.task.answer,
                rubric=self.config.task.rubric,
                notebook=notebook,
                proposed_solution=answer,
            )

            try:
                for attempt in range(self.config.rubric_attempts):
                    try:
                        response = await litellm.acompletion(
                            model=self.config.rubric_model,
                            api_base=self.config.rubric_model_api_base,
                            api_key=self.config.rubric_model_api_key,
                            messages=[{"role": "user", "content": prompt}],
                            timeout=self.config.rubric_timeout,
                            max_tokens=self.config.rubric_max_tokens,
                            max_retries=0,
                        )
                        break
                    except Exception:
                        if attempt == self.config.rubric_attempts - 1:
                            raise
                        logger.exception("Rubric model call failed, retrying.")
                        await asyncio.sleep(10)

                response_text = response.choices[0].message.content or ""
                matches = re.findall(r"<score>\s*(\d+)\s*</score>", response_text, flags=re.IGNORECASE)
                if not matches:
                    raise ValueError("Rubric response did not contain an integer <score>...</score>")

                score = int(matches[-1])
                max_points = self.config.task.max_points
                if max_points <= 0:
                    raise ValueError(f"Task max_points must be positive, got {max_points}")
                if score > max_points:
                    raise ValueError(f"Rubric score {score} exceeds max_points {max_points}")

                reward = score / max_points
                if self.config.log_dir is not None:
                    self.config.log_dir.mkdir(parents=True, exist_ok=True)
                    (self.config.log_dir / "nb.md").write_text(notebook)
                    (self.config.log_dir / "judge.txt").write_text(response_text)
                    (self.config.log_dir / "reward.json").write_text(
                        json.dumps(
                            {
                                "score": score,
                                "max_points": max_points,
                                "reward": reward,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    (self.config.log_dir / "judge_error.json").unlink(missing_ok=True)

                self._submitted = True

                return "Incorrect answer!" if reward < 1.0 else "Correct answer!"
            except Exception as error:
                logger.exception("Rubric model call failed.")
                if self.config.log_dir is not None:
                    self.config.log_dir.mkdir(parents=True, exist_ok=True)
                    (self.config.log_dir / "judge_error.json").write_text(
                        json.dumps({"status": "judge_failed", "error_type": type(error).__name__}) + "\n"
                    )
                return "Internal server error while judging the submitted answer."


def main() -> None:
    MCPServer(tyro.cli(MCPConfig, prog="bbh-mcp-server")).start()


if __name__ == "__main__":
    main()
