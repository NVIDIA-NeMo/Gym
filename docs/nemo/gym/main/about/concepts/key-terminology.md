"""
Production-grade implementation of NeMo Gym environment components.

This module provides a robust, type-safe implementation of the core concepts
defined in the NeMo Gym framework: Environment, Dataset, Agent Harness,
Verifier, State, and related components.

Requirements: https://docs.nvidia.com/nemo/gym/main/about/concepts/key-terminology/
"""

from __future__ import annotations

import abc
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
import json
import os
import signal
import sys
import traceback

# Configure module-level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class NeMoGymError(Exception):
    """Base exception for all NeMo Gym errors."""
    pass


class StateError(NeMoGymError):
    """Raised when state operations fail."""
    pass


class VerifierError(NeMoGymError):
    """Raised when verification/scoring fails."""
    pass


class HarnessError(NeMoGymError):
    """Raised when agent harness operations fail."""
    pass


class EpisodeError(NeMoGymError):
    """Raised when episode execution fails."""
    pass


class ConfigurationError(NeMoGymError):
    """Raised when configuration is invalid."""
    pass


class TimeoutException(NeMoGymError):
    """Raised when an operation times out."""
    pass


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

T_Action = TypeVar("T_Action", contravariant=True)
T_Observation = TypeVar("T_Observation", covariant=True)
T_Reward = TypeVar("T_Reward", covariant=True)
T_State = TypeVar("T_State", covariant=True)
T_Task = TypeVar("T_Task")
T_Metadata = TypeVar("T_Metadata")
T_Config = TypeVar("T_Config")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EpisodeStatus(enum.Enum):
    """Represents the outcome of an episode."""
    SUCCESS = "success"
    FAILURE = "failure"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    ERROR = "error"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


class TurnType(enum.Enum):
    """Defines the type of interaction in a turn."""
    USER_INPUT = "user_input"
    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM_MESSAGE = "system_message"

    def __str__(self) -> str:
        return self.value


class LogLevel(enum.IntEnum):
    """Custom log levels for NeMo Gym."""
    TRACE = 5
    VERBOSE = 15
    STANDARD = 20
    DETAILED = 25


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Step(Generic[T_Action, T_Observation, T_Reward]):
    """
    A single micro-level execution cycle inside the agent's harness.

    Represents the model generating a single tool call, executing it,
    and receiving the observation/response.

    Attributes:
        action: The action taken by the agent
        observation: The observation received from the environment
        reward: The reward signal for this step
        timestamp: Unix timestamp when the step was created
        metadata: Additional metadata for the step
    """

    action: T_Action
    observation: T_Observation
    reward: T_Reward
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate step data after initialization."""
        if self.reward is None:
            raise ValueError("Reward cannot be None")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"Metadata must be a dict, got {type(self.metadata)}")
        if self.timestamp <= 0:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to a serializable dictionary."""
        return {
            "action": self._serialize(self.action),
            "observation": self._serialize(self.observation),
            "reward": self._serialize(self.reward),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Serialize a value for JSON conversion."""
        if isinstance(value, enum.Enum):
            return value.value
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value


@dataclass(frozen=True)
class Turn(Generic[T_Action, T_Observation, T_Reward]):
    """
    A single round-trip interaction boundary between the external world
    (user/environment) and the agent within an episode.

    Attributes:
        turn_id: Unique identifier for the turn within an episode
        turn_type: Type of interaction
        steps: Tuple of steps in this turn
        timestamp: Unix timestamp when the turn was created
        metadata: Additional metadata for the turn
    """

    turn_id: int
    turn_type: TurnType
    steps: Tuple[Step[T_Action, T_Observation, T_Reward], ...]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate turn data after initialization."""
        if self.turn_id < 0:
            raise ValueError(f"turn_id must be non-negative, got {self.turn_id}")
        if not self.steps:
            raise ValueError("A turn must contain at least one step")
        if not isinstance(self.turn_type, TurnType):
            raise TypeError(f"turn_type must be TurnType, got {type(self.turn_type)}")
        if self.timestamp <= 0:
            raise ValueError(f"Invalid timestamp: {self.timestamp}")

    @property
    def total_reward(self) -> float:
        """Calculate the total reward for this turn."""
        try:
            return sum(
                float(step.reward) if step.reward is not None else 0.0
                for step in self.steps
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to calculate total reward: {e}")
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to a serializable dictionary."""
        return {
            "turn_id": self.turn_id,
            "turn_type": self.turn_type.value,
            "steps": [step.to_dict() for step in self.steps],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Episode(Generic[T_Action, T_Observation, T_Reward]):
    """
    A single attempt at completing a task (success / failure / turn limit).

    Contains the full trajectory/rollout/trace of the episode.

    Attributes:
        episode_id: Unique identifier for the episode
        task_id: Identifier of the task being solved
        status: Outcome status of the episode
        turns: Tuple of turns in this episode
        total_reward: Cumulative reward for the episode
        start_time: Unix timestamp when the episode started
        end_time: Unix timestamp when the episode ended (None if ongoing)
        metadata: Additional metadata for the episode
    """

    episode_id: str
    task_id: str
    status: EpisodeStatus
    turns: Tuple[Turn[T_Action, T_Observation, T_Reward], ...]
    total_reward: float
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate episode data after initialization."""
        if not self.episode_id or not self.episode_id.strip():
            raise ValueError("episode_id cannot be empty")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not isinstance(self.status, EpisodeStatus):
            raise TypeError(f"status must be EpisodeStatus, got {type(self.status)}")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) cannot be before start_time ({self.start_time})"
            )
        if self.start_time <= 0:
            raise ValueError(f"Invalid start_time: {self.start_time}")

    @property
    def duration_seconds(self) -> float:
        """Calculate the duration of the episode in seconds."""
        try:
            end = self.end_time or time.time()
            duration = end - self.start_time
            return max(0.0, duration)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to calculate duration: {e}")
            return 0.0

    @property
    def trajectory(self) -> List[Step[T_Action, T_Observation, T_Reward]]:
        """Return the flattened sequence of steps (state, action, reward)."""
        try:
            return [step for turn in self.turns for step in turn.steps]
        except Exception as e:
            logger.error(f"Failed to extract trajectory: {e}")
            return []

    @property
    def num_turns(self) -> int:
        """Return the number of turns in the episode."""
        return len(self.turns)

    @property
    def num_steps(self) -> int:
        """Return the total number of steps in the episode."""
        return len(self.trajectory)

    def is_completed(self) -> bool:
        """Check if the episode has completed."""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to a serializable dictionary."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "turns": [turn.to_dict() for turn in self.turns],
            "total_reward": self.total_reward,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save episode data to a JSON file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Episode saved to {filepath}")
        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to save episode to {filepath}: {e}")
            raise EpisodeError(f"Failed to save episode: {e}") from e

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> Episode:
        """Load episode data from a JSON file."""
        try:
            filepath = Path(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Note: This is a simplified reconstruction; full reconstruction
            # would require more complex deserialization logic
            return cls(
                episode_id=data["episode_id"],
                task_id=data["task_id"],
                status=EpisodeStatus(data["status"]),
                turns=tuple(
                    Turn(
                        turn_id=t["turn_id"],
                        turn_type=TurnType(t["turn_type"]),
                        steps=tuple(
                            Step(
                                action=s["action"],
                                observation=s["observation"],
                                reward=s["reward"],
                                timestamp=s.get("timestamp", time.time()),
                                metadata=s.get("metadata", {}),
                            )
                            for s in t["steps"]
                        ),
                        timestamp=t.get("timestamp", time.time()),
                        metadata=t.get("metadata", {}),
                    )
                    for t in data["turns"]
                ),
                total_reward=data["total_reward"],
                start_time=data.get("start_time", time.time()),
                end_time=data.get("end_time"),
                metadata=data.get("metadata", {}),
            )
        except (IOError, OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load episode from {filepath}: {e}")
            raise EpisodeError(f"Failed to load episode: {e}") from e


@dataclass(frozen=True)
class Task(Generic[T_Task, T_Metadata]):
    """
    A specific problem the agent must solve (e.g., "fix this GitHub bug",
    "answer this math question").

    Attributes:
        task_id: Unique identifier for the task
        prompt: The task prompt/description
        task_data: The actual task data
        metadata: Privileged information for verification
        max_turns: Maximum number of turns allowed
        max_steps_per_turn: Maximum steps per turn
        timeout_seconds: Maximum time allowed for the task
    """

    task_id: str
    prompt: str
    task_data: T_Task
    metadata: T_Metadata
    max_turns: int = 10
    max_steps_per_turn: int = 5
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate task configuration."""
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if self.max_steps_per_turn < 1:
            raise ValueError(
                f"max_steps_per_turn must be >= 1, got {self.max_steps_per_turn}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.timeout_seconds > 86400:  # 24 hours
            logger.warning(f"Unusually large timeout: {self.timeout_seconds}s")

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to a serializable dictionary."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "task_data": str(self.task_data),
            "metadata": str(self.metadata),
            "max_turns": self.max_turns,
            "max_steps_per_turn": self.max_steps_per_turn,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class Dataset(Generic[T_Task, T_Metadata]):
    """
    Collection of tasks, and metadata/privileged information necessary
    for verification.

    Attributes:
        dataset_id: Unique identifier for the dataset
        tasks: Tuple of tasks in the dataset
        version: Version string for the dataset
        metadata: Additional metadata for the dataset
    """

    dataset_id: str
    tasks: Tuple[Task[T_Task, T_Metadata], ...]
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataset configuration."""
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if not self.tasks:
            raise ValueError("Dataset must contain at least one task")
        if not self.version or not self.version.strip():
            raise ValueError("version cannot be empty")
        # Validate version format (semantic versioning)
        parts = self.version.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Invalid version format: {self.version}")

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, index: int) -> Task[T_Task, T_Metadata]:
        """Access a task by index."""
        try:
            return self.tasks[index]
        except IndexError as e:
            raise IndexError(
                f"Task index {index} out of range for dataset with {len(self.tasks)} tasks"
            ) from e

    def __iter__(self):
        """Iterate over tasks in the dataset."""
        return iter(self.tasks)

    def get_task_by_id(self, task_id: str) -> Optional[Task[T_Task, T_Metadata]]:
        """Find a task by its ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def shuffle(self, seed: Optional[int] = None) -> Dataset[T_Task, T_Metadata]:
        """Return a new dataset with tasks in random order."""
        import random

        rng = random.Random(seed)
        shuffled_tasks = list(self.tasks)
        rng.shuffle(shuffled_tasks)
        return Dataset(
            dataset_id=self.dataset_id,
            tasks=tuple(shuffled_tasks),
            version=self.version,
            metadata=self.metadata,
        )

    def split(
        self, train_ratio: float = 0.8, seed: Optional[int] = None
    ) -> Tuple[Dataset[T_Task, T_Metadata], Dataset[T_Task, T_Metadata]]:
        """Split the dataset into training and validation sets."""
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        import random

        rng = random.Random(seed)
        tasks = list(self.tasks)
        rng.shuffle(tasks)

        split_idx = int(len(tasks) * train_ratio)
        train_tasks = tuple(tasks