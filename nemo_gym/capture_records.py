import fcntl
from os import fsync
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import orjson
from pydantic import BaseModel, model_validator

from nemo_gym import RESULTS_DIR
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


class CallCaptureConfig(BaseModel):
    """Run-wide model-call capture settings from Gym's global config."""

    should_capture_calls: bool = False
    call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_capture_dir(self) -> "CallCaptureConfig":
        if not self.should_capture_calls:
            return self

        if self.call_capture_dir is None:
            raise ValueError("call_capture_dir is required when should_capture_calls=true")

        if not self.call_capture_dir.is_absolute():
            self.call_capture_dir = RESULTS_DIR / self.call_capture_dir

        return self


class BaseRolloutRecord(BaseModel):
    # Rollout ID
    rollout_id: str


class ModelCallRecord(BaseRolloutRecord):
    """Observability record derived from one captured model-server exchange."""

    # HTTP information
    status_code: int
    route: str

    # Timing information
    timestamp_start: float
    timestamp_end: float

    # Gym information
    model_ref: ModelServerRef

    # Model-call record
    request: "NeMoGymResponseCreateParamsNonStreaming"
    response: Optional["NeMoGymResponse"]  # Only present if the call succeeded
    error_response: Optional[str]  # Only present if the call failed

    # Raw information that is only logged if it differs from the standard request and response types
    # e.g. if it is the /v1/responses route, this will be None
    raw_request: Optional[Dict[str, Any]]
    # List[str] for streaming responses
    raw_response: Optional[Union[Dict[str, Any], List[str]]]


class AggregateModelCallRecords(BaseModel):
    # Any other typically useful aggregate information can be added here
    records: List[ModelCallRecord]

    @classmethod
    def from_records(cls, records: List[ModelCallRecord]) -> "AggregateModelCallRecords":
        return cls(records=records)


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{rollout_id}.capture.jsonl"

    def record(self, record: BaseRolloutRecord) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends across worker processes (a model server may run with
        ``num_workers > 1``, where the in-process lock can't coordinate); the in-process lock
        serializes threads. This does blocking file IO + fsync, so callers run it off the event
        loop (the capture middleware offloads it via ``asyncio.to_thread``).
        """
        line = orjson.dumps(record.model_dump(), default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(record.rollout_id)
        with self._lock:
            with path.open("ab") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, rollout_id: str) -> List[BaseRolloutRecord]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []

        exchanges: List[BaseRolloutRecord] = []
        # Stream line-by-line; a capture can be large (token-ids / logprobs).
        with self._lock:
            with path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        exchanges.append(ModelCallRecord.model_validate(orjson.loads(stripped)))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return exchanges

    def clear(self) -> None:
        rmtree(self.root, ignore_errors=True)

    def aggregate(self, rollout_id: str) -> AggregateModelCallRecords:
        records = self.read(rollout_id)
        return AggregateModelCallRecords.from_records(records)
