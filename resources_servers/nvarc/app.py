# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
NVARC Resource Server — self-contained ARC-AGI environment.

Supports two agent modes:
- transductive: model outputs grid directly (parsed from \\boxed{} or text)
- inductive: model outputs Python code with transform() function (executed in sandbox)

Zero external dependencies beyond nemo_gym + pydantic.
"""

import asyncio
import json
import re
import sys
from typing import List, Optional

from fastapi import FastAPI
from problem import Board, ColorPalette
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# =============================================================================
# Subprocess sandbox template (from progressive_learning/arc_agi/templates/python_subprocess.jinja)
# =============================================================================

SUBPROCESS_TEMPLATE = '''
import sys
import json
import io
import signal

def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, dict):
        return {{k: _convert_numpy_types(v) for k, v in obj.items()}}
    return obj

TRANSFORM_TIMEOUT_SECONDS = {timeout_seconds}

class TransformTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TransformTimeoutError(f"Transform execution exceeded {{TRANSFORM_TIMEOUT_SECONDS}}s")

_BANNED_BUILTINS = frozenset({{
    'open', 'input', 'breakpoint', 'help', 'license', 'credits', 'copyright',
    'exit', 'quit', 'vars', 'dir', 'globals', 'locals',
}})

_BANNED_MODULES = frozenset({{
    'os', 'subprocess', 'shutil', 'pathlib', 'builtins',
    'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
    'pickle', 'shelve', 'marshal', 'importlib', 'pkgutil',
    'ctypes', 'multiprocessing', 'threading', 'signal',
    'tempfile', 'fileinput', 'codecs', 'pty', 'fcntl',
    'resource', 'syslog', 'asyncio', 'concurrent',
}})

_original_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    base_module = name.split('.')[0]
    if base_module in _BANNED_MODULES:
        raise ImportError(f"Import of '{{name}}' is not allowed in sandbox")
    return _original_import(name, globals, locals, fromlist, level)

if isinstance(__builtins__, dict):
    _safe_builtins = {{k: v for k, v in __builtins__.items() if k not in _BANNED_BUILTINS}}
else:
    _safe_builtins = {{k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _BANNED_BUILTINS and not k.startswith('_')}}
    _safe_builtins['__name__'] = '__main__'
    _safe_builtins['__doc__'] = None

_safe_builtins['__import__'] = _restricted_import
_safe_builtins['__builtins__'] = _safe_builtins

_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    exec_globals = {{'__builtins__': _safe_builtins}}
    exec({content_repr}, exec_globals)

    if 'transform' not in exec_globals:
        raise ValueError("No 'transform' function defined in code")

    input_grid = {input_json}

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRANSFORM_TIMEOUT_SECONDS)

    try:
        result = exec_globals['transform'](input_grid)
    finally:
        signal.alarm(0)

    if result is None:
        sys.stdout = _original_stdout
        print(json.dumps({{"success": True, "result": None}}))
    else:
        if hasattr(result, 'detach') and hasattr(result, 'cpu'):
            result = result.detach().cpu().tolist()
        result = _convert_numpy_types(result)
        sys.stdout = _original_stdout
        print(json.dumps({{"success": True, "result": result}}))

except TransformTimeoutError as e:
    sys.stdout = _original_stdout
    print(json.dumps({{"success": False, "error": f"TimeoutError: {{str(e)}}"}})  )

except Exception as e:
    sys.stdout = _original_stdout
    print(json.dumps({{"success": False, "error": f"{{type(e).__name__}}: {{str(e)[:500]}}"}})  )
'''


# =============================================================================
# Request / Response models
# =============================================================================


class NVARCResourcesServerConfig(BaseResourcesServerConfig):
    agent_mode: str = "transductive"
    python_timeout_seconds: int = 30


class NVARCRunRequest(BaseRunRequest):
    train: List[dict] = Field(default_factory=list)
    test_input: List[List[int]] = Field(default_factory=list)
    expected_output: List[List[int]] = Field(default_factory=list)
    task_id: Optional[str] = None
    agent_mode: Optional[str] = None


class NVARCVerifyRequest(NVARCRunRequest, BaseVerifyRequest):
    pass


class NVARCVerifyResponse(BaseVerifyResponse):
    expected_output: List[List[int]]
    predicted_output: Optional[List[List[int]]] = None
    extraction_successful: bool = False
    exact_match: bool = False
    agent_mode: str = ""


# =============================================================================
# Server
# =============================================================================


class NVARCResourcesServer(SimpleResourcesServer):
    config: NVARCResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: NVARCVerifyRequest) -> NVARCVerifyResponse:
        agent_mode = body.agent_mode or self.config.agent_mode
        response_text = _extract_assistant_text(body)

        if agent_mode == "inductive":
            predicted = await self._verify_inductive(response_text, body.test_input)
        else:
            predicted = _parse_grid(response_text)

        extraction_successful = predicted is not None
        exact_match = extraction_successful and predicted == body.expected_output
        reward = 1.0 if exact_match else 0.0

        body_dict = body.model_dump()
        body_dict.pop("expected_output", None)
        body_dict.pop("agent_mode", None)
        return NVARCVerifyResponse(
            **body_dict,
            reward=reward,
            expected_output=body.expected_output,
            predicted_output=predicted,
            extraction_successful=extraction_successful,
            exact_match=exact_match,
            agent_mode=agent_mode,
        )

    async def _verify_inductive(
        self,
        response_text: str,
        test_input: List[List[int]],
    ) -> Optional[List[List[int]]]:
        code = _extract_python_code(response_text)
        if code is None:
            return None
        return await _execute_python(
            code,
            test_input,
            self.config.python_timeout_seconds,
        )


# =============================================================================
# Parsing helpers
# =============================================================================


def _extract_assistant_text(body: BaseVerifyRequest) -> str:
    texts = []
    for output in body.response.output:
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_grid(text: str) -> Optional[List[List[int]]]:
    """Parse grid from response using Board.from_text().

    Board.from_text() handles \\boxed{} extraction, space/comma-separated grids,
    and color palette mapping.
    """
    text = _strip_thinking(text)
    try:
        board = Board.from_text(text, color_palette=ColorPalette.integers())
        if board.is_valid:
            return board.board
    except (ValueError, AttributeError, IndexError):
        pass
    return None


def _extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks."""
    text = _strip_thinking(text)

    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    if "def transform" in text:
        return text.strip()

    return None


async def _execute_python(
    code: str,
    input_grid: List[List[int]],
    timeout_seconds: int = 30,
) -> Optional[List[List[int]]]:
    """Execute Python code in a sandboxed subprocess and return the output grid."""
    script = SUBPROCESS_TEMPLATE.format(
        timeout_seconds=timeout_seconds,
        content_repr=repr(code),
        input_json=json.dumps(input_grid),
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds + 5)

        if proc.returncode != 0:
            return None

        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return None

        result = json.loads(output)
        if result.get("success") and result.get("result") is not None:
            board = Board(board=result["result"])
            if board.is_valid:
                return board.board

    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        pass

    return None


if __name__ == "__main__":
    NVARCResourcesServer.run_webserver()
