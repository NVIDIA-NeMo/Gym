# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict
from typing import Dict, Optional, Tuple

import httpx

from .proof_utils import (
    determine_proof_status,
)
from ..utils import get_logger_name, python_doc_to_cmd_help

LOG = logging.getLogger(get_logger_name(__file__))


class Sandbox(abc.ABC):
    """Code execution sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        # Create async HTTP client with high limits
        self.http_session = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=2048, max_connections=2048),
        )
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        self.session_histories = defaultdict(list)  # session_id -> list of generated_code

    async def close(self):
        """Close the HTTP session."""
        await self.http_session.aclose()

    async def _send_request(self, request, timeout):
        session_id = request.pop("session_id", None)
        extra_headers = {}
        if session_id is not None:
            extra_headers["X-Session-ID"] = str(session_id)

        if self.ssh_server and self.ssh_key_path:
            # For SSH tunneling, use threads since there's no async version
            import sshtunnel_requests

            def ssh_request():
                sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
                return sshtunnel_request.post(
                    url=self._get_execute_url(),
                    data=json.dumps(request),
                    timeout=timeout + 5.0,
                    headers={"Content-Type": "application/json", **extra_headers},
                )

            # Native async requires more lines of code, so we use to_thread
            # Should be ok since this is a debug mode
            output = await asyncio.to_thread(ssh_request)
        else:
            output = await self.http_session.post(
                url=self._get_execute_url(),
                content=json.dumps(request),
                timeout=timeout + 5.0,
                headers={"Content-Type": "application/json", **extra_headers},
            )
        # retrying 502 errors
        if output.status_code == 502:
            raise httpx.TimeoutException("502 error")
        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output):
        pass

    @abc.abstractmethod
    def _get_execute_url(self):
        pass

    @abc.abstractmethod
    def _prepare_request(
        self,
        generated_code,
        timeout,
        language="ipython",
        std_input="",
        max_output_characters=1000,
        traceback_verbosity="Plain",
    ):
        pass

    @abc.abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a remote execution session if supported by the backend."""
        pass

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = "ipython",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
        traceback_verbosity="plain",  # could be plain, context, verbose, or minimal
    ) -> Tuple[Dict, str]:
        traceback_verbosity = traceback_verbosity.capitalize()
        if language in ["python", "pypy3", "python3", "lean4", "shell"] and session_id is not None:
            raise RuntimeError(
                f"Stateful execution for {language} is not supported. session_id is {session_id} but should be None"
            )
        if language not in ["ipython", "python", "pypy3", "python3", "lean4", "shell"]:
            raise ValueError(f"Unsupported language: {language}")
        if language != "ipython" and traceback_verbosity != "Plain":
            raise ValueError("Configurable traceback_verbosity is only supported for ipython")

        session_id_str = str(session_id) if session_id is not None else None

        request_session_id = session_id
        if request_session_id is None and language == "ipython":  # creating a new session with empty state
            request_session_id = uuid.uuid4()

        request_session_id_str = str(request_session_id) if request_session_id is not None else None

        TO_EXECUTE = generated_code
        request = self._prepare_request(
            TO_EXECUTE, timeout, language, std_input, max_output_characters, traceback_verbosity
        )
        request["session_id"] = request_session_id_str
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            output = {"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"}
        new_session_created = output.pop("new_session_created", False)

        # Rebuild state by re-executing history first, then execute the new code.
        # NOTE: Only cells that completed successfully are stored, so we intentionally omit re-running cells that errored
        # or timed out. This means restoration **can diverge** from the original interactive session in those cases, but
        # avoids re-triggering side effects from failing cells while keeping the replay simple.
        if session_id is not None and new_session_created and output.get("process_status") != "timeout":
            history = list(self.session_histories.get(session_id_str, []))
            if request_session_id_str is not None:
                try:
                    await self.delete_session(request_session_id_str)
                except Exception as exc:
                    LOG.warning(
                        "Failed to delete restarted session %s before restoration: %s",
                        request_session_id_str,
                        exc,
                    )
                self.session_histories[request_session_id_str] = history
            if history:
                # Restore session state by executing each history cell sequentially to preserve semantics
                for cell_index, cell_code in enumerate(history, start=1):
                    restore_request = self._prepare_request(
                        cell_code, timeout, language, std_input, max_output_characters, traceback_verbosity
                    )
                    restore_request["session_id"] = request_session_id_str
                    try:
                        restore_output = await self._send_request(restore_request, timeout)
                    except httpx.TimeoutException:
                        restore_output = {"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"}

                    if restore_output.get("process_status") != "completed":
                        LOG.error(
                            "Sandbox state restoration failed for session %s while replaying cell %d/%d with output: %s",
                            session_id,
                            cell_index,
                            len(history),
                            restore_output,
                        )
                        # Best-effort cleanup of the remote session so the next execution starts fresh.
                        if request_session_id_str is not None:
                            try:
                                await self.delete_session(request_session_id_str)
                            except Exception as exc:
                                LOG.warning(
                                    "Failed to delete session %s after restoration failure: %s",
                                    request_session_id_str,
                                    exc,
                                )

                        stderr_parts = (
                            "RuntimeError: Sandbox state restoration failed after the execution worker restarted. "
                            "The interactive session history has been cleared; please re-run the last code block without relying on prior state."
                        )
                        failure_output = {
                            "process_status": "error",
                            "stdout": "",
                            "stderr": stderr_parts + "\n",
                        }

                        return failure_output, request_session_id

            # Execute the new code once restoration has succeeded (or there was no history to replay)
            exec_request = self._prepare_request(
                generated_code, timeout, language, std_input, max_output_characters, traceback_verbosity
            )
            exec_request["session_id"] = request_session_id_str
            try:
                output = await self._send_request(exec_request, timeout)
            except httpx.TimeoutException:
                output = {"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"}

        # Append to history if successful execution (process_status == 'completed')
        if output.get("process_status") == "completed" and request_session_id_str is not None:
            self.session_histories[request_session_id_str].append(generated_code)

        output.pop("new_session_created", None)

        return output, request_session_id

    async def is_proof_correct(self, pred_output, timeout=30.0):
        TO_EXECUTE = pred_output

        request = self._prepare_request(TO_EXECUTE, timeout, "lean4")
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            return "timeout"
        return determine_proof_status(output)

    def _check_ready(self, timeout: float = 5.0) -> bool:
        """Readiness check against the sandbox health endpoint."""
        url = f"http://{self.host}:{self.port}/health"

        try:
            if self.ssh_server and self.ssh_key_path:
                import sshtunnel_requests

                sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
                response = sshtunnel_request.get(url=url, timeout=timeout)
            else:
                with httpx.Client() as client:
                    response = client.get(url=url, timeout=timeout)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def wait_for_sandbox(self, timeout: int = 5):
        while not self._check_ready(timeout=timeout):
            time.sleep(1)


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self):
        return f"http://{self.host}:{self.port}/execute"

    def _parse_request_output(self, output):
        try:
            return output.json()
        except json.JSONDecodeError:
            LOG.error("Error during parsing output: %s", output.text)
            return {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

    def _prepare_request(
        self,
        generated_code,
        timeout,
        language="ipython",
        std_input="",
        max_output_characters=1000,
        traceback_verbosity="Plain",
    ):
        return {
            "generated_code": generated_code,
            "std_input": std_input,
            "timeout": timeout,
            "language": language,
            "max_output_characters": max_output_characters,
            "traceback_verbosity": traceback_verbosity,
        }

    async def delete_session(self, session_id: str) -> None:
        """Delete an IPython session on the local sandbox server."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = await self.http_session.delete(
                    url=f"http://{self.host}:{self.port}/sessions/{session_id}",
                    timeout=10.0,
                    headers={"X-Session-ID": session_id},
                )
                if response.status_code == 200:  # Success
                    if session_id in self.session_histories:
                        del self.session_histories[session_id]
                    return
                if response.status_code == 404:  # We were routed to a different worker
                    LOG.warning(f"Session {session_id} not found (already deleted?). Treating as success.")
                    if session_id in self.session_histories:
                        del self.session_histories[session_id]
                    return
                response.raise_for_status()
            except (
                httpx.ReadTimeout,  # retry for other communication errors and statuses
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as e:
                LOG.warning("Retry %d/%d deleting session %s – %s", attempt + 1, max_retries, session_id, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    LOG.warning(f"Failed to delete session {session_id} after {max_retries} attempts. ")
            except Exception as e:
                LOG.warning(
                    "Failed to delete session %s: %s (type: %s, repr: %r)\nTraceback:\n%s",
                    session_id,
                    e,
                    type(e).__name__,
                    e,
                    traceback.format_exc(),
                )
                raise  # Re-raise unexpected exceptions


sandboxes = {
    "local": LocalSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f"\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}"
    return python_doc_to_cmd_help(Sandbox, docs_prefix=prefix, arg_prefix="sandbox.")
