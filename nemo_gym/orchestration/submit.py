import sys

import rich
from omegaconf import OmegaConf

from nemo_gym.decorators import experimental
from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.slurm import SlurmExecutor

_EXECUTORS = {
    SlurmComputeConfig: SlurmExecutor,
}


def _load_submit_config() -> tuple[SubmitConfig, bool]:
    """Load the submit config YAML and apply +key=value overrides from the command line."""
    config_path: str | None = None
    dry_run: bool = False
    override_tokens: list[str] = []

    for token in sys.argv[1:]:
        stripped = token.lstrip("+")
        if stripped.startswith("_submit_config="):
            config_path = stripped[len("_submit_config="):]
        elif stripped == "_dry_run=true":
            dry_run = True
        else:
            override_tokens.append(stripped)

    if config_path is None:
        rich.print("[red]error:[/red] No config provided. Use [bold]--config <path>[/bold] to specify a submit config YAML.")
        sys.exit(1)

    base = OmegaConf.load(config_path)
    overrides = OmegaConf.from_dotlist(override_tokens) if override_tokens else OmegaConf.create()
    merged = OmegaConf.merge(base, overrides)
    return SubmitConfig.model_validate(OmegaConf.to_container(merged, resolve=True)), dry_run


@experimental
def submit() -> None:  # pragma: no cover
    config, dry_run = _load_submit_config()
    compute = next(iter(config.compute.values()))
    _EXECUTORS[type(compute)]().run(config, dry_run=dry_run)
