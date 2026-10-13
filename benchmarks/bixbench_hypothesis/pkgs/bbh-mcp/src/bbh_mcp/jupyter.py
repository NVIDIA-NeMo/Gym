import asyncio
import inspect
import json
import logging
import secrets
import socket
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import nbformat
from websockets.asyncio.client import connect

from bbh_mcp.tool_utils import list_dir_tool

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JupyterKernel:
    id: str
    _sessions: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @staticmethod
    async def start(server: "JupyterServer", name: str, path: str | None = None) -> "JupyterKernel":
        kernel_options = {"name": name}
        if path is not None:
            kernel_options["path"] = path

        response = await server._request("POST", "/api/kernels", json=kernel_options)
        response.raise_for_status()

        kernel_id = response.json().get("id")
        if not isinstance(kernel_id, str):
            raise TypeError("Jupyter Server returned no kernel ID")

        return JupyterKernel(id=kernel_id)

    async def connect(self, server: "JupyterServer", session_id: str | None = None) -> str:
        async with self._execution_lock:
            if server._host is None or server._port is None or server._token is None:
                raise RuntimeError("Jupyter Server is not initialized")

            if session_id is None and self._sessions:
                session_id = next(iter(self._sessions))
            elif session_id is None:
                session_id = str(uuid.uuid4())

            if session_id in self._sessions:
                return session_id

            websocket_url = (
                f"ws://{server._host}:{server._port}/api/kernels/{self.id}/channels"
                f"?session_id={session_id}&token={server._token}"
            )
            self._sessions[session_id] = await connect(websocket_url)
            return session_id

    async def close(self, session_id: str):
        async with self._execution_lock:
            websocket = self._sessions.pop(session_id, None)
            if websocket is None:
                return

            try:
                await websocket.close()
            except RuntimeError as error:
                if "different loop" not in str(error):
                    raise
                websocket.transport.close()

    async def execute(
        self,
        server: "JupyterServer",
        code: str,
        session_id: str | None = None,
        timeout: float | None = None,
        store_history: bool = True,
    ) -> dict[str, Any]:
        async with self._execution_lock:
            if not isinstance(code, str):
                raise TypeError("code must be a string")

            if session_id is None:
                if len(self._sessions) != 1:
                    raise RuntimeError("session_id is required unless exactly one session exists")
                session_id = next(iter(self._sessions))

            websocket = self._sessions.get(session_id)
            if websocket is None:
                raise RuntimeError(f"Jupyter session is not initialized: {session_id}")

            message_id = str(uuid.uuid4())
            await websocket.send(
                json.dumps(
                    {
                        "channel": "shell",
                        "header": {
                            "msg_id": message_id,
                            "username": "bbh",
                            "session": session_id,
                            "date": datetime.now(UTC).isoformat(),
                            "msg_type": "execute_request",
                            "version": "5.3",
                        },
                        "parent_header": {},
                        "metadata": {},
                        "content": {
                            "code": code,
                            "silent": False,
                            "store_history": store_history,
                            "user_expressions": {},
                            "allow_stdin": False,
                            "stop_on_error": True,
                        },
                    },
                )
            )

            outputs: list[dict[str, Any]] = []
            reply: dict[str, Any] | None = None
            idle = False
            try:
                async with asyncio.timeout(timeout):
                    while not (idle and reply is not None):
                        message = json.loads(await websocket.recv())
                        if message.get("parent_header", {}).get("msg_id") != message_id:
                            continue

                        message_type = message.get("header", {}).get("msg_type")
                        content = message.get("content", {})

                        if message_type in {"stream", "execute_result", "display_data", "error"}:
                            outputs.append({"output_type": message_type, **content})
                        elif message_type == "execute_reply":
                            reply = content
                        elif message_type == "status" and content.get("execution_state") == "idle":
                            idle = True
            except TimeoutError:
                response = await server._request("POST", f"/api/kernels/{self.id}/interrupt")
                response.raise_for_status()
                raise

            return {
                "outputs": outputs,
                "error": bool(reply and reply.get("status") == "error")
                or any(output["output_type"] == "error" for output in outputs),
                "execution_count": reply.get("execution_count") if reply else None,
            }

    async def stop(self, server: "JupyterServer"):
        for session_id in list(self._sessions):
            await self.close(session_id)

        response = await server._request("DELETE", f"/api/kernels/{self.id}")
        response.raise_for_status()


class JupyterServer:
    def __init__(
        self,
        backend: Literal["local", "docker"],
        container_image: str | None = None,
        container_mounts: list[tuple[str | Path, str | Path, bool]] | None = None,
        kernel_workdir: str | Path = "/app",
    ):
        if backend not in {"local", "docker"}:
            raise ValueError(f"unsupported Jupyter backend: {backend}")

        if backend == "docker" and not container_image:
            raise ValueError("docker_image must be set for the docker backend")

        self.backend = backend
        self.container_image = container_image
        self.kernel_workdir = Path(kernel_workdir).expanduser()
        self.container_mounts = [
            (Path(host_path), str(container_path), read_only)
            for host_path, container_path, read_only in (container_mounts or [])
        ]

        ## Server Bookkeeping.
        self._process: asyncio.subprocess.Process | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._token: str | None = None
        self._xsrf_token: str | None = None
        self._http_client: httpx.AsyncClient | None = None

        ## Kernel Bookkeeping.
        self._kernels: dict[str, JupyterKernel] = {}

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._http_client is None:
            raise RuntimeError("Jupyter Server HTTP client is not initialized")

        if self._token is None:
            raise RuntimeError("Jupyter Server token is not initialized")

        request_params = dict(params or {})
        request_params["token"] = self._token

        request_headers = dict(headers or {})
        request_headers["Authorization"] = f"token {self._token}"
        if self._xsrf_token is not None:
            request_headers["X-XSRFToken"] = self._xsrf_token

        return await self._http_client.request(
            method,
            path,
            params=request_params,
            headers=request_headers,
            **kwargs,
        )

    async def start(self, host: str | None = None, port: int | None = None, timeout: int = 30):
        if self._process is not None:
            assert self._host is not None
            assert self._port is not None
            assert self._token is not None
            assert self._xsrf_token is not None
            assert self._http_client is not None
            return

        if host is None:
            host = "127.0.0.1"

        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
                socket_handle.bind((host, 0))
                port = socket_handle.getsockname()[1]

        token = secrets.token_urlsafe(32)

        command = [
            "jupyter",
            "server",
            "--no-browser",
            "--allow-root",
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.log_level=WARN",
            f"--ServerApp.root_dir={self.kernel_workdir}",
            "--LabApp.log_level=WARN",
            f"--IdentityProvider.token={token}",
        ]

        if self.backend == "local":
            command += [
                f"--ip={host}",
                f"--port={port}",
            ]
        elif self.backend == "docker":
            docker_command = ["docker", "run", "--rm", "-p", f"{host}:{port}:8888"]
            for host_path, container_path, read_only in self.container_mounts:
                mode = ":ro" if read_only else ""
                docker_command += ["-v", f"{host_path}:{container_path}{mode}"]

            command = (
                docker_command
                + [self.container_image]
                + command
                + [
                    "--ip=0.0.0.0",
                    "--port=8888",
                ]
            )
        else:
            raise NotImplementedError(f"Unsupported backend {self.backend}")

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._host = host
        self._port = port
        self._token = token
        self._http_client = httpx.AsyncClient(base_url=f"http://{host}:{port}")

        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self._process.returncode is not None:
                stdout, stderr = await self._process.communicate()
                stdout_text = stdout.decode(errors="replace") if stdout else ""
                stderr_text = stderr.decode(errors="replace") if stderr else ""
                raise RuntimeError(
                    f"Jupyter Server exited with code {self._process.returncode}\n"
                    f"stdout:\n{stdout_text}\n"
                    f"stderr:\n{stderr_text}"
                )

            try:
                await self._request("GET", "/tree", timeout=2)
                response = await self._request("GET", "/api", timeout=2)
            except httpx.HTTPError:
                await asyncio.sleep(0.25)
                continue

            if response.is_success:
                self._xsrf_token = self._http_client.cookies.get("_xsrf")
                if self._xsrf_token is not None:
                    return

            await asyncio.sleep(0.25)

        raise TimeoutError(f"Jupyter Server did not become ready within {timeout} seconds")

    async def start_kernel(self, name: Literal["ir", "python3"]) -> tuple[str, str]:
        kernel = await JupyterKernel.start(
            self,
            name,
            path=str(self.kernel_workdir),
        )
        self._kernels[kernel.id] = kernel
        session_id = await kernel.connect(self)
        return kernel.id, session_id

    def get_kernel(self, kernel_id: str) -> JupyterKernel:
        kernel = self._kernels.get(kernel_id)
        if kernel is None:
            raise RuntimeError(f"Jupyter kernel is not registered: {kernel_id}")
        return kernel

    async def stop_kernel(self, kernel_id: str):
        kernel = self._kernels.pop(kernel_id, None)
        if kernel is None:
            return

        await kernel.stop(self)

    async def stop(self):
        if self._process is None:
            return

        process = self._process
        http_client = self._http_client
        self._process = None
        self._http_client = None
        self._port = None

        if http_client is not None:
            await http_client.aclose()

        if process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()


class JupyterNotebook:
    def __init__(self):
        self._nb = nbformat.v4.new_notebook()

    def __len__(self) -> int:
        return len(self._nb.cells)

    def add(
        self,
        source: str,
        outputs: list[dict[str, Any]],
        execution_count: int | None,
        cell_idx: int | None,
    ) -> tuple[int, nbformat.NotebookNode]:
        notebook_outputs = [
            {key: value for key, value in output.items() if key not in {"transient"}} for output in outputs
        ]

        if cell_idx is not None:
            cell = self._nb.cells[cell_idx]
            cell.source = source
            cell.outputs = notebook_outputs
            cell.execution_count = execution_count
            return cell_idx, cell

        cell = nbformat.v4.new_code_cell(
            source=source,
            outputs=notebook_outputs,
            execution_count=execution_count,
        )
        self._nb.cells.append(cell)
        return len(self._nb.cells) - 1, cell

    def render_cell_output(self, cell_idx: int, show_images: bool = False) -> str:
        cell = self._nb.cells[cell_idx]

        rendered: list[str] = []
        for output in cell.outputs:
            output_type = output.get("output_type")
            if output_type == "stream":
                rendered.append(str(output.get("text", "")))
            elif output_type in {"execute_result", "display_data"}:
                data = output.get("data", {})

                if "text/plain" in data:
                    rendered.append(str(data.pop("text/plain")))

                for mime_type, value in data.items():
                    if mime_type.startswith("image/"):
                        if not show_images:
                            rendered.append("[Image generated, not shown]")
                        elif mime_type == "image/svg+xml":
                            rendered.append(str(value))
                        else:
                            rendered.append(f"![{mime_type}](data:{mime_type};base64,{value})")
                    elif mime_type == "application/json":
                        rendered.append(json.dumps(value, default=str))
                    else:
                        rendered.append(str(value))
            elif output_type == "error":
                traceback = output.get("traceback", [])
                if traceback:
                    rendered.append("\n".join(str(line) for line in traceback))
                else:
                    rendered.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
        return "".join(rendered)

    def render(self, language: str | None = None, show_images: bool = False) -> str:
        md: list[str] = []
        for cell_idx, cell in enumerate(self._nb.cells):
            md.extend([f"## Cell {cell_idx}", f"```{language or ''}", cell.source, "```\n"])
            md.extend(
                [f"### Output {cell_idx}", "```", self.render_cell_output(cell_idx, show_images=show_images), "```\n"]
            )

        return "\n".join(md).strip()


class JupyterClient:
    """Communicate with a single kernel managed by a :class:`JupyterServer`."""

    def __init__(self, server: JupyterServer, kernel_name: str, cell_timeout: float = 3600.0) -> None:
        self.server = server
        self.cell_timeout = cell_timeout
        self.kernel_name = kernel_name

        ## Bookkeeping.
        self.notebook: JupyterNotebook | None = None
        self._kernel_id: str | None = None
        self._meta_kernel_id: str | None = None

    async def run_cell(self, code: str, idx: int | None = None) -> str:
        if not isinstance(code, str):
            code = str(code)

        try:
            idx = int(idx)
        except (TypeError, ValueError, OverflowError):
            idx = None

        if idx is not None and (idx < 0 or idx >= len(self.notebook)):
            idx = None

        if self._kernel_id is None:
            await self.reset_kernel()

        kernel = self.server.get_kernel(self._kernel_id)
        try:
            result = await kernel.execute(self.server, code, timeout=self.cell_timeout)
        except TimeoutError:
            logger.exception("Kernel execution timed out.")
            return f"TimeoutError: Code execution timed out after {self.cell_timeout} seconds"

        cell_idx, _ = self.notebook.add(
            source=code,
            outputs=result["outputs"],
            execution_count=result["execution_count"],
            cell_idx=idx,
        )

        return f"[Cell #{cell_idx}]\n" + self.notebook.render_cell_output(cell_idx)

    async def reset_kernel(self):
        """Delete the current kernel and create a fresh one."""
        await self.server.stop_kernel(self._kernel_id)
        self._kernel_id, _ = await self.server.start_kernel(self.kernel_name)
        self.notebook = JupyterNotebook()

    async def list_dir(self, directory: str, max_files: int, show_hidden: bool) -> str:
        """Implements it via Python code execution to avoid separate filesystem handling making it portable to any backend."""

        if self._meta_kernel_id is None:
            self._meta_kernel_id, _ = await self.server.start_kernel(name="python3")

        source = textwrap.dedent(inspect.getsource(list_dir_tool))
        code = source + textwrap.dedent(
            f"""
            try:
                print(list_dir_tool({directory!r}, {max_files!r}, {show_hidden!r}), end="")
            finally:
                del list_dir_tool
            """
        )

        kernel = self.server.get_kernel(self._meta_kernel_id)
        result = await kernel.execute(self.server, code, timeout=self.cell_timeout, store_history=False)

        return "".join(output.get("text", "") for output in result["outputs"] if output.get("output_type") == "stream")
