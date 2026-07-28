import sys

import rich
from omegaconf import OmegaConf

from nemo_gym.orchestration.api import SubmitConfig


def _load_submit_config() -> SubmitConfig:
    """Load the submit config YAML and apply +key=value overrides from the command line."""
    config_path: str | None = None
    override_tokens: list[str] = []

    for token in sys.argv[1:]:
        stripped = token.lstrip("+")
        if stripped.startswith("_submit_config="):
            config_path = stripped[len("_submit_config="):]
        else:
            override_tokens.append(stripped)

    if config_path is None:
        rich.print("[red]error:[/red] No config provided. Use [bold]--config <path>[/bold] to specify a submit config YAML.")
        sys.exit(1)

    base = OmegaConf.load(config_path)
    overrides = OmegaConf.from_dotlist(override_tokens) if override_tokens else OmegaConf.create()
    merged = OmegaConf.merge(base, overrides)
    return SubmitConfig.model_validate(OmegaConf.to_container(merged, resolve=True))


def submit() -> None:  # pragma: no cover
    _load_submit_config()
    rich.print("[bold]gym eval submit[/bold]: not yet implemented.")
