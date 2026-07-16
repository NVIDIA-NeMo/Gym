from pathlib import Path
from typing import Optional

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import ConfigPathNotFoundError


def resolve_input_path(input_path: str | Path, error_msg: Optional[str] = None) -> Path:
    _input_path = Path(input_path)
    if not _input_path.is_absolute():
        _cwd_path = Path.cwd() / input_path
        _input_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / input_path
    if not _input_path.is_file():
        error_msg = error_msg or f"Given input file not found: '{input_path}'. Check it is spelled correctly and exists."
        raise ConfigPathNotFoundError(error_msg)
    return _input_path


def failures_path_for(output_fpath: Path) -> Path:
    """Sidecar failures path derived from a rollout output path."""
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")