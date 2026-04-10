from pathlib import Path
from subprocess import run


def clone_and_checkout():
    cwd = Path(__file__).parent
    repo_path = cwd / "tau2-bench"
    if not repo_path.exists():
        run(
            """git clone https://github.com/bxyu-nvidia/tau2-bench""",
            shell=True,
            cwd=cwd,
            check=True,
            executable="/bin/bash",
        )

    run(
        """source .venv/bin/activate \
&& cd tau2-bench \
&& git checkout b76acee2e625b8fb22deeb63cb0a3e756d5f094e \
&& uv sync --active""",
        shell=True,
        cwd=cwd,
        check=True,
        executable="/bin/bash",
    )


def prepare_data() -> Path:
    cwd = Path(__file__).parent
    repo_path = cwd / "tau2-bench"
    assert repo_path.exists()

    run(
        """source .venv/bin/activate \
&& cd tau2-bench \
&& current_commit=$(git rev-parse HEAD) \
&& git checkout bxyu/nemo_gym_data \
&& bash dump_nemo_gym_data.sh \
&& git checkout $current_commit""",
        shell=True,
        cwd=cwd,
        check=True,
        executable="/bin/bash",
    )

    return repo_path / "nemo_gym_data"
