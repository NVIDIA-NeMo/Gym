from asyncio import FIRST_COMPLETED, create_subprocess_shell, create_task, wait
from asyncio.subprocess import PIPE
from pathlib import Path
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
        process = await create_subprocess_shell(command, stdout=PIPE, stderr=PIPE)

        output_lines = []
        log_file = await aiofiles.open(self.log_file, "a")
        while True:
            tasks = [
                create_task(process.stdout.readline()),
                create_task(process.stderr.readline()),
            ]
            done, _ = await wait(tasks, return_when=FIRST_COMPLETED)
            line = done[0].result()

            if not line:
                break  # EOF reached

            decoded_line = line.decode().rstrip()

            print(decoded_line)
            output_lines.append(decoded_line)
            await log_file.write(decoded_line + "\n")

        stdout_rem, stderr = await process.communicate()
        output_lines.append(stdout_rem.decode().rstrip())
        output_lines.append(stderr.decode().rstrip())

        await log_file.write(output_lines[-2] + "\n" + output_lines[-1] + "\n")
        await log_file.close()

        return CommandOutput(
            returncode=process.returncode,
            output_lines=output_lines,
        )

    async def _preflight_check_git_repo(self, repo: str, branch: str) -> str:
        """
        Returns the head sha
        """

        command_output = await self._run_command(
            command=f'git ls-remote --exit-code "{repo}" "refs/heads/{branch}" | cut -f1'
        )
        assert command_output.returncode == 0, (
            f"Preflight git repo check failed for {repo} @ {branch}. See {self.log_file}"
        )


class MooncakeBuilder(BaseBuilder):
    config: MooncakeConfig

    def preflight_checks(self) -> None:
        return super().preflight_checks()


class BuildContainerHelper:
    config: BuildContainerConfig

    def run(self) -> None:
        pass
