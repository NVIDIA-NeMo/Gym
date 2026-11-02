from asyncio import create_subprocess_shell
from asyncio.subprocess import PIPE, STDOUT
from os import killpg
from pathlib import Path
from signal import SIGKILL
from typing import List, Optional

import aiofiles
from pydantic import BaseModel

from nemo_gym.config_types import BaseNeMoGymCLIConfig


class SlurmConfig(BaseNeMoGymCLIConfig):
    pass


class MooncakeConfig(BaseNeMoGymCLIConfig):
    pass


class BuildContainerConfig(BaseNeMoGymCLIConfig):
    input_container_fpath: str
    output_container_fpath: str
    log_file: Path

    slurm: Optional[SlurmConfig] = None
    mooncake: Optional[MooncakeConfig] = None


class CommandOutput(BaseModel):
    returncode: int
    output_lines: List[str]


class BaseBuilder(BaseModel):
    config: BaseNeMoGymCLIConfig

    # Log file is separate from the config so that we dont have the duplicate this log file property on every config.
    log_file: Path

    async def perform_preflight_checks(self) -> None:
        raise NotImplementedError

    def get_command_str(self) -> str:
        raise NotImplementedError

    async def _run_command(self, command: str) -> CommandOutput:
        async with aiofiles.open(self.log_file, "a") as log_file:
            process = await create_subprocess_shell(command, stdout=PIPE, stderr=STDOUT, start_new_session=True)
            output_lines = []

            try:
                assert process.stdout is not None

                async for line in process.stdout:
                    line = line.decode(errors="replace").rstrip()

                    print(line, flush=True)
                    output_lines.append(line)
                    await log_file.write(line + "\n")

                returncode = await process.wait()
            except:
                try:
                    killpg(process.pid, SIGKILL)
                except ProcessLookupError:
                    pass  # Already exited.

                await process.communicate()
                raise

        return CommandOutput(
            returncode=returncode,
            output_lines=output_lines,
        )

    async def _preflight_check_git_repo(self, repo: str, branch: str) -> str:
        """
        Returns the head sha
        """

        command_output = await self._run_command(command=f'git ls-remote --exit-code "{repo}" "refs/heads/{branch}"')

        error_message = f"Preflight git repo check failed for {repo} @ {branch}. See {self.log_file}"
        assert command_output.returncode == 0 and command_output.output_lines, error_message

        last_output_split = command_output.output_lines[-1].split()
        assert len(last_output_split) == 2 and last_output_split[1] == f"refs/heads/{branch}", error_message

        return last_output_split[0]


class MooncakeBuilder(BaseBuilder):
    config: MooncakeConfig


class BuildContainerHelper:
    config: BuildContainerConfig

    def run(self) -> None:
        pass
