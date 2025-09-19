from os import environ
from pathlib import Path
from subprocess import Popen


def _setup_env_command(dir_path: Path) -> str:  # pragma: no cover
    return f"""cd {dir_path} \\
    && uv venv --allow-existing \\
    && source .venv/bin/activate \\
    && uv pip install -r requirements.txt \\
   """


def _run_command(command: str, working_directory: Path) -> Popen:  # pragma: no cover
    custom_env = environ.copy()
    custom_env["PYTHONPATH"] = f"{working_directory.absolute()}:{custom_env.get('PYTHONPATH', '')}"
    print(f"Executing command:\n{command}\n")
    return Popen(command, executable="/bin/bash", shell=True, env=custom_env)
