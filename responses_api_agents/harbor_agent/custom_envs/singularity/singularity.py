"""
Singularity/Apptainer environment for running tasks on HPC clusters.

This environment converts Docker images to Singularity .sif format and
runs a FastAPI server inside the container to handle command execution.
"""

import asyncio
import asyncio.subprocess
import fcntl
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import httpx

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


class MemoryLimitExceededError(Exception):
    """Raised when a container exceeds its memory limit."""
    pass


class SingularityEnvironment(BaseEnvironment):
    """
    Singularity-based environment for HPC clusters.

    This environment:
    1. Pulls Docker images and converts them to .sif format
    2. Runs a FastAPI server inside the container for command execution
    3. Uses bind mounts for file transfer

    Optional kwargs:
        singularity_image_cache_dir: Path to cache .sif files (defaults to a temp directory)
        singularity_force_pull: Force re-conversion of Docker images to .sif
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_force_pull: bool | str = False,
        *args,
        **kwargs,
    ):
        # Extract singularity-specific kwargs before calling super().__init__
        if singularity_image_cache_dir:
            self._image_cache_dir = Path(singularity_image_cache_dir)
        else:
            self._image_cache_dir = Path(tempfile.mkdtemp(prefix="singularity_cache_"))
        # Handle string "true"/"false" from CLI --ek flag
        self._force_pull = singularity_force_pull in (True, "true", "True", "1", "yes")

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )

        self._server_process: asyncio.subprocess.Process | None = None
        self._server_port: int | None = None
        self._staging_dir: Path | None = None
        self._sif_path: Path | None = None
        self._stream_task: asyncio.Task | None = None
        self._memory_watchdog_task: asyncio.Task | None = None
        self._http_client: httpx.AsyncClient | None = None

        # Memory limit from task config (in bytes for easy comparison)
        self._memory_limit_bytes = self.task_env_config.memory_mb * 1024 * 1024
        # Flag set when watchdog kills container - checked in exec() to raise proper error
        self._memory_limit_exceeded: str | None = None

        # Get the working directory from Dockerfile
        self._workdir = self._get_workdir_from_dockerfile()

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        """Validate that required files and configuration exist."""
        # Must have either docker_image in config or a Dockerfile
        if not self.task_env_config.docker_image and not self._dockerfile_path.exists():
            raise FileNotFoundError(
                f"Singularity environment requires either 'docker_image' in task.toml "
                f"or a Dockerfile at {self._dockerfile_path}"
            )

    def _get_workdir_from_dockerfile(self) -> str:
        """Extract WORKDIR from Dockerfile if present.

        Returns the LAST WORKDIR directive, since that's the effective one.
        """
        if not self._dockerfile_path.exists():
            return "/app"

        workdir = "/app"  # Default
        try:
            with open(self._dockerfile_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith("WORKDIR "):
                        workdir = line.split(None, 1)[1].strip()
        except Exception:
            pass

        return workdir

    def _reserve_port(self) -> tuple[socket.socket, int]:
        """Reserve a free port by keeping the socket bound.

        Returns a tuple of (socket, port). The caller must close the socket
        when ready to use the port, minimizing the race condition window.

        Uses SO_REUSEADDR so the port can be immediately reused after closing.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))  # Bind to loopback, random port
        s.listen(1)
        port = s.getsockname()[1]
        return s, port

    def _get_image_name(self) -> str:
        """Get the Docker image name to convert."""
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image

        raise ValueError(
            "No docker_image specified in task.toml. "
            "Singularity requires pre-built images."
        )

    async def _convert_docker_to_sif(self, docker_image: str) -> Path:
        """Convert a Docker image to Singularity .sif format.

        Uses file locking to prevent race conditions when multiple concurrent
        tasks try to convert the same image simultaneously.
        """
        # Create safe filename
        safe_name = docker_image.replace("/", "_").replace(":", "_")
        sif_path = self._image_cache_dir / f"{safe_name}.sif"
        lock_path = self._image_cache_dir / f"{safe_name}.sif.lock"

        # Create cache directory if needed
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

        # Quick check before acquiring lock (optimization for cached images)
        if not self._force_pull and sif_path.exists():
            self.logger.debug(f"Using cached Singularity image: {sif_path}")
            return sif_path

        # Acquire file lock to prevent concurrent conversions
        self.logger.debug(f"Acquiring lock for image conversion: {docker_image}")
        lock_file = open(lock_path, "w")
        try:
            # Run blocking flock in thread executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            )
            self.logger.debug(f"Lock acquired for: {docker_image}")

            # Handle force pull - delete existing .sif if requested
            if self._force_pull and sif_path.exists():
                self.logger.debug(f"Force pull enabled, removing cached image: {sif_path}")
                sif_path.unlink()

            # Double-check after acquiring lock (another process may have created it)
            if sif_path.exists():
                self.logger.debug(
                    f"Using cached Singularity image (created by another process): {sif_path}"
                )
                return sif_path

            self.logger.info(f"Converting Docker image to Singularity: {docker_image}")

            # Ensure image has a tag
            if ":" not in docker_image:
                docker_image = f"{docker_image}:latest"

            # Use a temporary file for pulling, then rename atomically
            tmp_sif_path = self._image_cache_dir / f"{safe_name}.sif.tmp.{self.session_id}"

            # Pull from Docker registry
            cmd = ["singularity", "pull", str(tmp_sif_path), f"docker://{docker_image}"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                # Clean up failed temporary file
                if tmp_sif_path.exists():
                    tmp_sif_path.unlink()
                error_msg = stderr.decode(errors="replace")
                raise RuntimeError(f"Failed to convert Docker image: {error_msg}")

            # Atomically rename temp file to final path
            tmp_sif_path.rename(sif_path)

            self.logger.info(f"Created Singularity image: {sif_path}")
            return sif_path
        finally:
            # Release the lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    async def _start_server(self) -> None:
        """Start the FastAPI server inside the Singularity container.

        Uses port reservation with retry logic to handle race conditions where
        another process might grab the port between reservation and binding.
        """
        # Create staging directory for file transfers (done once, outside retry loop)
        self._staging_dir = Path(tempfile.mkdtemp(prefix="singularity_staging_"))
        self._staging_dir.chmod(0o755)

        # Copy server.py to staging with a non-obvious name to avoid being killed
        server_script = Path(__file__).parent / "server.py"
        staging_server = self._staging_dir / "_hbexec.py"
        shutil.copy(server_script, staging_server)

        # Create a bootstrap script to protect the server from kill commands.
        # Agents often run commands like:
        #   - "pkill python3" / "killall python3" - kills all python processes
        #   - "pkill -f server.py" - kills processes matching "server.py"
        #
        # We protect against these by:
        # 1. Creating a symlink to python3 named "harbor-exec" so the process name
        #    in /proc/PID/comm is "harbor-exec", not "python3"
        # 2. Using a non-obvious script name "_hbexec.py" that agents won't target
        bootstrap_script = self._staging_dir / "bootstrap.sh"
        bootstrap_script.write_text(
            "#!/bin/bash\n"
            "# Harbor server bootstrap - protects against agent kill commands\n"
            '# Process name will be "harbor-exec" (not "python3")\n'
            '# Script name is "_hbexec.py" (not "server.py" or "app.py")\n'
            'ln -sf "$(which python3)" /tmp/harbor-exec\n'
            'exec /tmp/harbor-exec "$@"\n'
        )
        bootstrap_script.chmod(0o755)

        # Try to start server with retry logic for port conflicts
        max_port_retries = 3
        last_error = None

        for port_attempt in range(max_port_retries):
            # Reserve a port - keeps socket bound to minimize race window
            reserved_socket, port = self._reserve_port()
            self._server_port = port

            # Build singularity command
            # Note: --memory and --cpus flags are NOT used because they require cgroups
            # support (systemd running as init), which is typically not available on HPC
            # clusters. Resource limits should be enforced at the SLURM level instead
            # (via --mem, --cpus-per-task in sbatch/srun).
            cmd = [
                "singularity", "run",
                "--pwd", self._workdir,
                "--writable-tmpfs",
                "--fakeroot",
                "--containall",
                # Bind mounts
                "-B", f"{self._staging_dir}:/staging",
                "-B", f"{self.trial_paths.verifier_dir}:{EnvironmentPaths.verifier_dir}",
                "-B", f"{self.trial_paths.agent_dir}:{EnvironmentPaths.agent_dir}",
                str(self._sif_path),
                "/staging/bootstrap.sh", "/staging/_hbexec.py",
                "--port", str(self._server_port),
                "--workdir", self._workdir,
            ]

            self.logger.info(
                f"Starting Singularity container with server on port {self._server_port} "
                f"(attempt {port_attempt + 1}/{max_port_retries})"
            )

            # Release the reserved port and immediately start the container
            # The small window here is unavoidable, but SO_REUSEADDR helps
            reserved_socket.close()

            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Start background task to stream server output
            self._stream_task = asyncio.create_task(self._stream_server_output())

            # Wait for server to be ready
            self._http_client = httpx.AsyncClient(timeout=30.0)
            server_ready = False

            for _ in range(60):  # 60 second timeout for server startup
                try:
                    response = await self._http_client.get(
                        f"http://localhost:{self._server_port}/health"
                    )
                    if response.status_code == 200:
                        # IMPORTANT: Verify OUR server is still alive before declaring ready.
                        # Another concurrent trial may have grabbed this port and responded
                        # to our health check (Singularity shares host network namespace).
                        if self._server_process.returncode is not None:
                            await self._stream_task
                            last_error = RuntimeError(
                                f"Port collision on {self._server_port}: health check succeeded "
                                f"but our server process died. Another trial grabbed this port."
                            )
                            self.logger.warning(
                                f"Health check succeeded but server process died - "
                                f"port {self._server_port} collision with another trial"
                            )
                            break  # Will trigger retry with new port
                        self.logger.info("Singularity FastAPI server is ready")
                        # Start memory watchdog now that server is ready
                        self._memory_watchdog_task = asyncio.create_task(
                            self._memory_watchdog()
                        )
                        server_ready = True
                        break
                except httpx.RequestError:
                    pass

                # Check if process died (possibly due to port conflict)
                if self._server_process.returncode is not None:
                    await self._stream_task
                    last_error = RuntimeError(
                        f"Server process died on port {self._server_port}. "
                        f"Check trial.log for server output."
                    )
                    self.logger.warning(
                        f"Server failed to start on port {self._server_port}, "
                        "will retry with new port"
                    )
                    break

                await asyncio.sleep(1)

            if server_ready:
                return

            # Clean up failed attempt before retry
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

        # All retries exhausted
        raise last_error or RuntimeError(
            f"Failed to start Singularity FastAPI server after {max_port_retries} port attempts"
        )

    async def _stream_server_output(self) -> None:
        """Stream server stdout/stderr to logger in real-time."""
        if not self._server_process or not self._server_process.stdout:
            return

        try:
            async for line in self._server_process.stdout:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    # Log at debug level to avoid cluttering trial logs
                    self.logger.debug(f"[server] {decoded}")
        except Exception as e:
            self.logger.debug(f"Server output stream ended: {e}")

    def _get_process_tree_memory(self, pid: int) -> int:
        """Get total PSS memory of a process and all its descendants.

        Uses PSS (Proportional Set Size) which properly accounts for shared memory
        by dividing it proportionally among sharing processes. Falls back to RSS
        if PSS is unavailable. Directly reads /proc for efficiency (no subprocess).
        Returns memory in bytes, or 0 if unable to read.

        Note: /proc reads are essentially instantaneous (kernel memory, not disk)
        so this doesn't need to be async.
        """

        def get_all_descendants(root_pid: int) -> set[int]:
            """Get all PIDs in the process tree by walking /proc children."""
            pids = set()
            to_visit = [root_pid]

            while to_visit:
                current = to_visit.pop()
                if current in pids:
                    continue
                pids.add(current)

                # Find children via /proc/[pid]/task/*/children
                try:
                    task_dir = Path(f"/proc/{current}/task")
                    for tid_dir in task_dir.iterdir():
                        children_file = tid_dir / "children"
                        try:
                            for child in children_file.read_text().split():
                                if child.isdigit():
                                    to_visit.append(int(child))
                        except (OSError, PermissionError):
                            pass
                except (OSError, PermissionError):
                    pass

            return pids

        def get_process_memory(p: int) -> int:
            """Get memory for a single process: PSS if available, else RSS."""
            # Try PSS from smaps_rollup (Linux 4.14+, more accurate)
            try:
                for line in Path(f"/proc/{p}/smaps_rollup").read_text().splitlines():
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            # Fallback to RSS from statm (second field, in pages)
            try:
                rss_pages = int(Path(f"/proc/{p}/statm").read_text().split()[1])
                return rss_pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, PermissionError, ValueError, IndexError):
                pass

            return 0

        try:
            return sum(get_process_memory(p) for p in get_all_descendants(pid))
        except Exception:
            return 0

    async def _memory_watchdog(self) -> None:
        """Monitor memory usage and kill container if it exceeds the limit.

        This runs as a background task while the container is active.
        Features:
        - Adaptive intervals: checks every 1s when memory >50%, every 3s otherwise
        - Explosion detection: warns if memory growth rate would hit limit in <5s
        - Kill threshold at 95%: leaves headroom before actual OOM
        """
        # Configuration
        base_interval = 3  # seconds - normal check interval
        fast_interval = 1  # seconds - when memory is high
        warning_threshold = 0.5  # Switch to fast mode at 50%
        kill_threshold = 0.95  # Kill at 95% to leave headroom

        # State tracking
        last_mem_usage = 0
        last_check_time = 0.0

        self.logger.debug(
            f"Memory watchdog started: limit={self._memory_limit_bytes // 1024 // 1024}MB, "
            f"kill_at={kill_threshold*100:.0f}%, intervals={fast_interval}s/{base_interval}s"
        )

        while True:
            try:
                # Check if server is still running
                if not self._server_process or self._server_process.returncode is not None:
                    self.logger.debug("Memory watchdog: server process ended, stopping watchdog")
                    break

                # Get memory usage of the entire process tree
                current_time = asyncio.get_event_loop().time()
                mem_usage = self._get_process_tree_memory(self._server_process.pid)
                mem_mb = mem_usage / 1024 / 1024
                limit_mb = self._memory_limit_bytes / 1024 / 1024
                usage_pct = mem_usage / self._memory_limit_bytes if self._memory_limit_bytes > 0 else 0

                # Calculate growth rate (bytes per second)
                if last_check_time > 0 and last_mem_usage > 0:
                    time_delta = current_time - last_check_time
                    if time_delta > 0:
                        growth_rate = (mem_usage - last_mem_usage) / time_delta
                        # Warn if growth rate would hit limit in less than 5 seconds
                        if growth_rate > 0:
                            remaining_bytes = self._memory_limit_bytes * kill_threshold - mem_usage
                            time_to_limit = (
                                remaining_bytes / growth_rate if growth_rate > 0 else float("inf")
                            )
                            if 0 < time_to_limit < 5:
                                self.logger.warning(
                                    f"Memory explosion detected: {mem_mb:.0f}MB, "
                                    f"growing {growth_rate / 1024 / 1024:.0f}MB/s, "
                                    f"~{time_to_limit:.1f}s until limit"
                                )

                last_mem_usage = mem_usage
                last_check_time = current_time

                # Kill if exceeded threshold (95% to leave some headroom)
                if mem_usage > self._memory_limit_bytes * kill_threshold:
                    error_msg = (
                        "Container exceeded memory limit "
                        f"({mem_mb:.0f}MB > {limit_mb * kill_threshold:.0f}MB)"
                    )
                    self.logger.error(
                        f"Memory limit exceeded: {mem_mb:.0f}MB > {limit_mb * kill_threshold:.0f}MB "
                        f"({usage_pct*100:.0f}%). Killing container to prevent OOM."
                    )
                    # Set flag BEFORE killing so exec() can check it
                    self._memory_limit_exceeded = error_msg
                    # Kill the Singularity process - it will clean up internal processes
                    self._server_process.kill()
                    # Don't raise here - let exec() detect the error and raise properly
                    return

                # Adaptive interval: fast when memory is high
                if usage_pct > warning_threshold:
                    interval = fast_interval
                else:
                    interval = base_interval

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                self.logger.debug("Memory watchdog cancelled")
                raise
            except Exception as e:
                self.logger.debug(f"Memory watchdog error (continuing): {e}")
                await asyncio.sleep(base_interval)  # Sleep to prevent busy-loop on errors

    async def start(self, force_build: bool) -> None:
        """Start the Singularity environment."""
        # Get the Docker image to use
        docker_image = self._get_image_name()

        # Convert to .sif format
        self._sif_path = await self._convert_docker_to_sif(docker_image)

        # Start the FastAPI server
        await self._start_server()

        # Upload environment files to /app (for prebuilt images with shared category Dockerfiles)
        await self._upload_environment_files()

    async def _upload_environment_files(self) -> None:
        """Upload environment/files to /app in the container.

        For Singularity with prebuilt images, task-specific files are not baked
        into the image. This method uploads them at runtime after the container starts.
        """
        files_dir = self.environment_dir / "files"
        if not files_dir.exists():
            return

        # Check if there are any files to upload (skip empty directories)
        try:
            has_files = any(files_dir.iterdir())
        except (PermissionError, OSError):
            has_files = False

        if not has_files:
            return

        self.logger.debug(f"Uploading environment files from {files_dir} to /app")
        try:
            await self.upload_dir(
                source_dir=files_dir,
                target_dir="/app",
            )
        except Exception as e:
            self.logger.warning(f"Failed to upload environment files: {e}")

    async def stop(self, delete: bool) -> None:
        """Stop the Singularity environment and all child processes."""
        # Close HTTP client (don't send /shutdown - it could hit another trial's server
        # if there was a port collision; just terminate our process directly)
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        # Cancel memory watchdog first
        if self._memory_watchdog_task and not self._memory_watchdog_task.done():
            self._memory_watchdog_task.cancel()
            try:
                await self._memory_watchdog_task
            except asyncio.CancelledError:
                pass

        # Terminate server process and its children
        # Singularity with --containall should propagate signals to internal processes
        if self._server_process and self._server_process.returncode is None:
            pid = self._server_process.pid

            # First, try graceful termination
            # Singularity should propagate SIGTERM to container processes
            self._server_process.terminate()
            self.logger.debug(f"Sent SIGTERM to Singularity process {pid}")

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill
                self.logger.debug("Graceful shutdown timed out, force killing")
                self._server_process.kill()
                await self._server_process.wait()

            # Run pkill as a backup to catch any escaped child processes
            # (e.g., processes that daemonized or detached)
            try:
                subprocess.run(
                    ["pkill", "-9", "-P", str(pid)],
                    capture_output=True,
                    timeout=5
                )
            except Exception:
                pass

        # Cancel stream task if running
        if hasattr(self, "_stream_task") and self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Cleanup staging directory
        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)

        # Note: We don't delete .sif files as they can be reused
        if delete:
            self.logger.debug(
                f"Singularity image preserved at {self._sif_path} for reuse"
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command in the Singularity container via HTTP."""
        if not self._http_client or not self._server_port:
            raise RuntimeError("Singularity environment not started")

        # Check if memory watchdog already killed the container
        if self._memory_limit_exceeded:
            raise MemoryLimitExceededError(self._memory_limit_exceeded)

        try:
            # Calculate HTTP timeout:
            # - If timeout_sec provided: add 10s buffer for HTTP overhead
            # - If timeout_sec is None: no HTTP timeout (match Docker/Daytona behavior)
            http_timeout = timeout_sec + 10 if timeout_sec else None

            response = await self._http_client.post(
                f"http://localhost:{self._server_port}/exec",
                json={
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                },
                timeout=http_timeout,
            )
            response.raise_for_status()
            result = response.json()

            exec_result = ExecResult(
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                return_code=result.get("return_code", 1),
            )

            # Log errors so they're visible in trial logs (stderr is otherwise discarded)
            if exec_result.return_code != 0 and exec_result.stderr:
                self.logger.warning(
                    f"Command failed (rc={exec_result.return_code}): {exec_result.stderr}"
                )

            return exec_result

        except httpx.TimeoutException:
            # Check if memory watchdog killed the container during request
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            raise asyncio.TimeoutError(
                f"HTTP request timed out after {http_timeout} seconds"
                if http_timeout else "HTTP request timed out"
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            # Check if memory watchdog killed the container
            if self._memory_limit_exceeded:
                raise MemoryLimitExceededError(self._memory_limit_exceeded)
            raise  # Re-raise original error if not memory-related

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a file to the container via staging directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Copy to staging
        staging_file = self._staging_dir / source.name
        shutil.copy2(source, staging_file)

        # Move to target in container
        result = await self.exec(f"cp /staging/{source.name} {target_path}")
        if result.return_code != 0:
            raise RuntimeError(f"Failed to upload file: {result.stderr}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory to the container via staging directory."""
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        # Copy to staging
        staging_subdir = self._staging_dir / source.name
        if staging_subdir.exists():
            shutil.rmtree(staging_subdir)
        shutil.copytree(source, staging_subdir)

        # Create target directory and copy
        await self.exec(f"mkdir -p {target_dir}")
        result = await self.exec(f"cp -r /staging/{source.name}/* {target_dir}/")
        if result.return_code != 0:
            # Try alternative approach
            result = await self.exec(f"cp -r /staging/{source.name} {target_dir}")
            if result.return_code != 0:
                raise RuntimeError(f"Failed to upload directory: {result.stderr}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the container via staging directory."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Copy to staging in container
        filename = Path(source_path).name
        staging_path = f"/staging/download_{filename}"
        result = await self.exec(f"cp {source_path} {staging_path}")
        if result.return_code != 0:
            raise RuntimeError(f"Failed to download file: {result.stderr}")

        # Copy from staging to target
        staging_file = self._staging_dir / f"download_{filename}"
        if staging_file.exists():
            shutil.copy2(staging_file, target)
            staging_file.unlink()
        else:
            raise RuntimeError(f"File not found in staging: {staging_file}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the container via staging directory."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        # Copy to staging in container
        dirname = Path(source_dir).name
        staging_path = f"/staging/download_{dirname}"
        result = await self.exec(f"cp -r {source_dir} {staging_path}")
        if result.return_code != 0:
            raise RuntimeError(f"Failed to download directory: {result.stderr}")

        # Copy from staging to target
        staging_subdir = self._staging_dir / f"download_{dirname}"
        if staging_subdir.exists():
            # Copy contents
            for item in staging_subdir.iterdir():
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(staging_subdir)
        else:
            raise RuntimeError(f"Directory not found in staging: {staging_subdir}")
