"""
Glossary of Key Terminology for Agent-Based Systems

This module defines the core concepts and types used in agent-based evaluation
and training systems, following the NVIDIA NeMo Gym terminology.

The module provides a comprehensive type system for defining:
- Tasks and datasets for agent evaluation
- Episodes, turns, and steps for tracking agent interactions
- Agent harnesses and solvers for managing agent-environment interaction
- Verifiers and scorers for evaluating agent performance
- Environments and benchmarks for reproducible evaluation
- Runtimes and sandboxes for execution isolation
"""

from __future__ import annotations

import abc
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

# Configure module-level logger with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T_State = TypeVar("T_State", covariant=True)
T_Action = TypeVar("T_Action", covariant=True)
T_Reward = TypeVar("T_Reward", covariant=True)
T_Task = TypeVar("T_Task")
T_Metadata = TypeVar("T_Metadata")
T_Observation = TypeVar("T_Observation")
T_Output = TypeVar("T_Output")
T_Config = TypeVar("T_Config")


# =============================================================================
# Custom Exceptions
# =============================================================================

class AgentSystemError(Exception):
    """Base exception for all agent system errors."""
    pass


class TaskError(AgentSystemError):
    """Raised when there's an error with task definition or execution."""
    pass


class EpisodeError(AgentSystemError):
    """Raised when there's an error during episode execution."""
    pass


class VerificationError(AgentSystemError):
    """Raised when verification fails."""
    pass


class RuntimeError(AgentSystemError):
    """Raised when there's a runtime execution error."""
    pass


class ConfigurationError(AgentSystemError):
    """Raised when configuration is invalid."""
    pass


# =============================================================================
# Enumerations
# =============================================================================

class EpisodeOutcome(enum.Enum):
    """Defines the possible outcomes of an episode/trial."""

    SUCCESS = "success"
    FAILURE = "failure"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    ERROR = "error"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"

    def is_terminal(self) -> bool:
        """Check if this outcome represents a terminal state."""
        return self in {
            EpisodeOutcome.SUCCESS,
            EpisodeOutcome.FAILURE,
            EpisodeOutcome.TURN_LIMIT_REACHED,
            EpisodeOutcome.TIMEOUT,
        }

    def is_success(self) -> bool:
        """Check if this outcome represents success."""
        return self == EpisodeOutcome.SUCCESS


class InteractionMode(enum.Enum):
    """Defines the mode of interaction between agent and environment."""

    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"
    MULTI_TURN = "multi_turn"

    @classmethod
    def from_string(cls, value: str) -> "InteractionMode":
        """Create InteractionMode from string, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ConfigurationError(f"Invalid interaction mode: {value}") from e


class VerifierType(enum.Enum):
    """Defines the type of verifier/scorer/grader."""

    RULE_BASED = "rule_based"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"

    def supports_batch_verification(self) -> bool:
        """Check if this verifier type supports batch verification."""
        return self in {VerifierType.RULE_BASED, VerifierType.HYBRID}


class SandboxType(enum.Enum):
    """Defines the type of sandbox for execution isolation."""

    DOCKER = "docker"
    APPTAINER = "apptainer"
    LOCAL_PROCESS = "local_process"
    KUBERNETES = "kubernetes"
    AWS_LAMBDA = "aws_lambda"


# =============================================================================
# Core Data Classes
# =============================================================================

@dataclass(frozen=True)
class Task(Generic[T_Task, T_Metadata]):
    """
    A specific problem the agent must solve.

    Tasks define the objective the agent is working toward, such as
    "fix this GitHub bug" or "answer this math question." Each task includes
    the initial conditions, success criteria, and any constraints on how
    the solution may be achieved.

    Attributes:
        task_id: Unique identifier for the task.
        description: Human-readable description of the task.
        initial_conditions: The starting state/context for the task.
        success_criteria: Criteria that define successful completion.
        constraints: Any constraints on how the solution may be achieved.
        metadata: Additional metadata and privileged information for verification.
        tags: Optional tags for categorizing the task.
        difficulty: Optional difficulty level (1-10).
        estimated_time: Optional estimated time to complete in seconds.
    """

    task_id: str
    description: str
    initial_conditions: T_Task
    success_criteria: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[T_Metadata] = None
    tags: Set[str] = field(default_factory=set)
    difficulty: Optional[int] = None
    estimated_time: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the task after initialization."""
        errors: List[str] = []

        if not self.task_id or not self.task_id.strip():
            errors.append("task_id must not be empty")
        if not self.description or not self.description.strip():
            errors.append("description must not be empty")
        if self.initial_conditions is None:
            errors.append("initial_conditions must not be None")
        if not self.success_criteria:
            errors.append("success_criteria must not be empty")
        if self.difficulty is not None and not (1 <= self.difficulty <= 10):
            errors.append("difficulty must be between 1 and 10")
        if self.estimated_time is not None and self.estimated_time <= 0:
            errors.append("estimated_time must be positive")

        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"Task validation failed for {self.task_id}: {error_msg}")
            raise TaskError(error_msg)

        logger.debug(f"Task {self.task_id} initialized successfully")

    def matches_tags(self, required_tags: Set[str]) -> bool:
        """Check if task has all required tags."""
        return required_tags.issubset(self.tags)


@dataclass(frozen=True)
class Dataset(Generic[T_Task, T_Metadata]):
    """
    A collection of tasks with metadata and privileged information.

    The dataset provides the ground truth or reference solutions that enable
    automated scoring of agent performance. Datasets may include multiple
    difficulty levels, task variants, or domain-specific categories.

    Attributes:
        dataset_id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        tasks: Collection of tasks in the dataset.
        version: Version string for the dataset.
        metadata: Additional dataset-level metadata.
        created_at: Timestamp when the dataset was created.
    """

    dataset_id: str
    name: str
    tasks: Sequence[Task[T_Task, T_Metadata]]
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate the dataset after initialization."""
        errors: List[str] = []

        if not self.dataset_id or not self.dataset_id.strip():
            errors.append("dataset_id must not be empty")
        if not self.name or not self.name.strip():
            errors.append("name must not be empty")
        if not self.tasks:
            errors.append("tasks must not be empty")

        # Validate version format
        import re
        if not re.match(r"^\d+\.\d+\.\d+$", self.version):
            errors.append(f"Invalid version format: {self.version}")

        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"Dataset validation failed for {self.dataset_id}: {error_msg}")
            raise ConfigurationError(error_msg)

        logger.info(
            f"Dataset {self.dataset_id} (v{self.version}) initialized "
            f"with {len(self.tasks)} tasks"
        )

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, index: int) -> Task[T_Task, T_Metadata]:
        """Get task by index."""
        if not isinstance(index, int):
            raise TypeError(f"Index must be int, got {type(index)}")
        if index < 0 or index >= len(self.tasks):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.tasks)} tasks")
        return self.tasks[index]

    def get_task(self, task_id: str) -> Optional[Task[T_Task, T_Metadata]]:
        """Retrieve a task by its ID."""
        if not task_id or not task_id.strip():
            raise ValueError("task_id must not be empty")

        for task in self.tasks:
            if task.task_id == task_id:
                return task

        logger.warning(f"Task {task_id} not found in dataset {self.dataset_id}")
        return None

    def filter_by_tags(self, required_tags: Set[str]) -> List[Task[T_Task, T_Metadata]]:
        """Filter tasks by required tags."""
        if not required_tags:
            return list(self.tasks)
        return [task for task in self.tasks if task.matches_tags(required_tags)]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        difficulties = [t.difficulty for t in self.tasks if t.difficulty is not None]
        return {
            "total_tasks": len(self.tasks),
            "avg_difficulty": sum(difficulties) / len(difficulties) if difficulties else None,
            "min_difficulty": min(difficulties) if difficulties else None,
            "max_difficulty": max(difficulties) if difficulties else None,
            "unique_tags": set().union(*[t.tags for t in self.tasks]),
        }


@dataclass
class Step(Generic[T_State, T_Action, T_Reward]):
    """
    A single, micro-level execution cycle inside the agent's harness.

    For example, the model generating a single tool call, executing that call,
    and receiving the observation or response. Steps are finer-grained than
    turns and represent the internal processing units of the agent's
    decision-making loop.

    Attributes:
        step_number: Sequential step number within the episode.
        state: The state before the action was taken.
        action: The action taken by the agent.
        reward: The reward received after the action.
        next_state: The state after the action was taken.
        observation: The observation received from the environment.
        metadata: Additional step-level metadata.
        duration: Duration of the step in seconds.
        timestamp: When the step was recorded.
    """

    step_number: int
    state: T_State
    action: T_Action
    reward: T_Reward
    next_state: Optional[T_State] = None
    observation: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate the step after initialization."""
        if not isinstance(self.step_number, int):
            raise TypeError(f"step_number must be int, got {type(self.step_number)}")
        if self.step_number < 0:
            raise ValueError(f"step_number must be non-negative, got {self.step_number}")
        if self.duration is not None and self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

        logger.debug(f"Step {self.step_number} recorded at {self.timestamp}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "observation": self.observation,
            "metadata": self.metadata,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Turn(Generic[T_State, T_Action, T_Reward]):
    """
    A single round-trip interaction boundary between the external world
    (user/environment) and the agent within an episode.

    A turn begins when the agent receives input from the environment and ends
    when the agent produces output that is sent back. Multiple turns may occur
    within a single episode.

    Attributes:
        turn_number: Sequential turn number within the episode.
        input_data: The input received from the environment.
        output_data: The output produced by the agent.
        steps: The sequence of steps within this turn.
        metadata: Additional turn-level metadata.
        duration: Duration of the turn in seconds.
        timestamp: When the turn was recorded.
    """

    turn_number: int
    input_data: Any
    output_data: Any
    steps: List[Step[T_State, T_Action, T_Reward]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate the turn after initialization."""
        if not isinstance(self.turn_number, int):
            raise TypeError(f"turn_number must be int, got {type(self.turn_number)}")
        if self.turn_number < 0:
            raise ValueError(f"turn_number must be non-negative, got {self.turn_number}")
        if self.duration is not None and self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

        logger.debug(
            f"Turn {self.turn_number} recorded with {len(self.steps)} steps "
            f"at {self.timestamp}"
        )

    def add_step(self, step: Step[T_State, T_Action, T_Reward]) -> None:
        """Add a step to this turn."""
        if not isinstance(step, Step):
            raise TypeError(f"step must be an instance of Step, got {type(step)}")
        if step.step_number != len(self.steps):
            logger.warning(
                f"Step number mismatch: expected {len(self.steps)}, got {step.step_number}"
            )
        self.steps.append(step)
        logger.debug(f"Step {step.step_number} added to turn {self.turn_number}")

    def get_total_reward(self) -> float:
        """Calculate total reward from all steps in this turn."""
        total = 0.0
        for step in self.steps:
            try:
                total += float(step.reward)
            except (TypeError, ValueError):
                logger.warning(f"Could not convert reward {step.reward} to float")
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Episode(Generic[T_State, T_Action, T_Reward]):
    """
    A single attempt at completing a task.

    An episode represents one complete trial where an agent attempts to solve
    a task. It tracks the sequence of turns, the final outcome, and all
    associated metadata.

    Attributes:
        episode_id: Unique identifier for the episode.
        task: The task being attempted.
        turns: Sequence of turns in this episode.
        outcome: The final outcome of the episode.
        total_reward: Cumulative reward across all turns.
        metadata: Additional episode-level metadata.
        max_turns: Maximum number of turns allowed.
        max_duration: Maximum duration allowed in seconds.
        start_time: When the episode started.
        end_time: When the episode ended.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: Optional[Task] = None
    turns: List[Turn[T_State, T_Action, T_Reward]] = field(default_factory=list)
    outcome: Optional[EpisodeOutcome] = None
    total_reward: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    max_turns: int = 100
    max_duration: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Validate the episode after initialization."""
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")
        if self.max_duration is not None and self.max_duration <= 0:
            raise ValueError(f"max_duration must be positive, got {self.max_duration}")

        logger.debug(f"Episode {self.episode_id} initialized")

    def add_turn(self, turn: Turn[T_State, T_Action, T_Reward]) -> None:
        """Add a turn to this episode."""
        if not isinstance(turn, Turn):
            raise TypeError