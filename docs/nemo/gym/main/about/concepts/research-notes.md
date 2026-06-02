"""
NeMo Gym Agent Framework - Production Implementation
====================================================
This module implements a high-quality agent framework following NeMo Gym's
key terminology and architecture patterns.

Core Concepts:
- Task: A well-defined problem instance the agent must solve
- Dataset: Collection of tasks with metadata for verification
- Episode: Single attempt at completing a task
- Turn: Single round-trip interaction boundary
- Step: Micro-level execution cycle inside agent's harness
- Trajectory: Sequence of (state, action, reward) tuples
- Model/Policy: Weights doing reasoning/generation
- Agent Harness/Solver: Defines rollout protocol
- Agent: Model + Harness composition
- Verifier/Scorer/Grader: Scores episodes for rewards
- State: Per-episode mutable state
- Environment: Everything except the model
- Benchmark: Fixed configuration for reproducible evaluation
- Runtime: Execution infrastructure
- Sandbox: Isolated runtime environment
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Callable,
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

# Type variables for generic components
T = TypeVar("T")
StateType = TypeVar("StateType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType", float, int)

# Configure module logger
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration of possible task statuses during execution."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED_SUCCESS = auto()
    COMPLETED_FAILURE = auto()
    TIMEOUT = auto()
    ERROR = auto()


class TurnLimitExceededError(Exception):
    """Raised when an episode exceeds the maximum number of allowed turns."""

    def __init__(self, max_turns: int, actual_turns: int):
        self.max_turns = max_turns
        self.actual_turns = actual_turns
        super().__init__(
            f"Episode exceeded maximum turns: {actual_turns} > {max_turns}"
        )


class InvalidTaskError(Exception):
    """Raised when a task fails validation checks."""

    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"Invalid task '{task_id}': {reason}")


class SandboxIsolationError(Exception):
    """Raised when sandbox isolation is compromised or unavailable."""

    def __init__(self, sandbox_id: str, message: str):
        self.sandbox_id = sandbox_id
        super().__init__(f"Sandbox '{sandbox_id}' isolation error: {message}")


@runtime_checkable
class Verifiable(Protocol):
    """Protocol for objects that can be verified/scored."""

    def verify(self, expected: Any, actual: Any) -> float:
        """Verify and return a score between 0 and 1."""
        ...


@dataclass(frozen=True)
class Task(Generic[T]):
    """
    A well-defined problem instance the agent must solve.

    Attributes:
        id: Unique identifier for the task
        prompt: Natural language or structured prompt describing the task
        metadata: Additional context and verification information
        success_criteria: Criteria for determining task completion
        max_turns: Maximum allowed turns for this task
        timeout_seconds: Maximum execution time in seconds
    """

    id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Optional[Dict[str, Any]] = None
    max_turns: int = 10
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.id or not self.id.strip():
            raise InvalidTaskError(self.id, "Task ID cannot be empty")
        if not self.prompt or not self.prompt.strip():
            raise InvalidTaskError(self.id, "Task prompt cannot be empty")
        if self.max_turns < 1:
            raise InvalidTaskError(self.id, "max_turns must be >= 1")
        if self.timeout_seconds <= 0:
            raise InvalidTaskError(self.id, "timeout_seconds must be > 0")


@dataclass
class Dataset(Generic[T]):
    """
    Collection of tasks with metadata and privileged information for verification.

    Attributes:
        name: Dataset name for identification
        tasks: List of tasks in the dataset
        version: Dataset version for reproducibility
        metadata: Additional dataset-level metadata
    """

    name: str
    tasks: List[Task[T]]
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if not self.tasks:
            raise ValueError("Dataset must contain at least one task")
        if not self.version:
            raise ValueError("Dataset version cannot be empty")

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, index: int) -> Task[T]:
        """Get task by index with bounds checking."""
        if not 0 <= index < len(self.tasks):
            raise IndexError(f"Task index {index} out of range [0, {len(self.tasks)})")
        return self.tasks[index]

    def get_task_by_id(self, task_id: str) -> Optional[Task[T]]:
        """Retrieve a task by its unique identifier."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        logger.warning(f"Task with id '{task_id}' not found in dataset '{self.name}'")
        return None


@dataclass
class State:
    """
    Per-episode state that changes as the agent or other actors take actions.

    Attributes:
        episode_id: Unique identifier for the episode
        task: The current task being executed
        turn_count: Number of turns completed
        step_count: Number of steps completed
        start_time: When the episode started
        last_update: When the state was last modified
        data: Arbitrary state data (file systems, databases, etc.)
        status: Current task status
    """

    episode_id: str
    task: Task
    turn_count: int = 0
    step_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING

    def update(self, **kwargs: Any) -> None:
        """Update state attributes and timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Attempted to set unknown state attribute: {key}")
        self.last_update = datetime.now()

    def is_expired(self, timeout_seconds: float) -> bool:
        """Check if the episode has exceeded its timeout."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed > timeout_seconds


@dataclass
class Turn:
    """
    A single round-trip interaction boundary between the external world
    and the agent within an episode.

    Attributes:
        turn_number: Sequential turn number within the episode
        input: Input received from the environment
        output: Response produced by the agent
        steps: List of steps executed within this turn
        start_time: When the turn started
        end_time: When the turn ended
    """

    turn_number: int
    input: Any
    output: Optional[Any] = None
    steps: List["Step"] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_step(self, step: "Step") -> None:
        """Add a step to this turn."""
        self.steps.append(step)

    def complete(self, output: Any) -> None:
        """Mark the turn as complete with the given output."""
        self.output = output
        self.end_time = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Return the duration of this turn in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class Step:
    """
    A single, micro-level execution cycle inside the agent's harness.

    Attributes:
        step_number: Sequential step number within the episode
        action: The action taken by the agent (e.g., tool call)
        observation: The observation received after the action
        reward: Immediate reward for this step
        start_time: When the step started
        end_time: When the step ended
        metadata: Additional step metadata
    """

    step_number: int
    action: Any
    observation: Optional[Any] = None
    reward: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, observation: Any, reward: float = 0.0) -> None:
        """Mark the step as complete with observation and reward."""
        self.observation = observation
        self.reward = reward
        self.end_time = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Return the duration of this step in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class Trajectory:
    """
    Data produced during an episode: the sequence of (state, action, reward) tuples.

    Attributes:
        episode_id: Unique identifier for the episode
        task: The task being executed
        turns: List of turns in the episode
        steps: List of steps in the episode
        final_reward: Final reward for the episode
        success: Whether the episode was successful
        metadata: Additional trajectory metadata
    """

    episode_id: str
    task: Task
    turns: List[Turn] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to the trajectory."""
        self.turns.append(turn)
        self.steps.extend(turn.steps)

    def add_step(self, step: Step) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)

    @property
    def total_reward(self) -> float:
        """Calculate the total reward across all steps."""
        return sum(step.reward for step in self.steps)

    @property
    def average_reward(self) -> float:
        """Calculate the average reward per step."""
        if not self.steps:
            return 0.0
        return self.total_reward / len(self.steps)


class Model(ABC, Generic[T]):
    """
    Abstract base class for models/policies that do reasoning/generation.

    Subclasses must implement the `generate` method.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Generate a response based on the given prompt and context.

        Args:
            prompt: The input prompt for generation
            context: Optional context information
            **kwargs: Additional generation parameters

        Returns:
            The generated response

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate()")


class AgentHarness(ABC, Generic[T]):
    """
    Abstract base class for agent harnesses/solvers.

    Defines the rollout protocol, e.g., single step, multi-step, multi-turn.
    May include tools, context management, turn limits, etc.
    """

    @abstractmethod
    async def run_episode(
        self,
        task: Task[T],
        model: Model[T],
        verifier: Optional[Verifier] = None,
        sandbox: Optional[Sandbox] = None,
    ) -> Trajectory:
        """
        Run a single episode for the given task.

        Args:
            task: The task to execute
            model: The model to use for generation
            verifier: Optional verifier for scoring
            sandbox: Optional sandbox for isolation

        Returns:
            The trajectory of the episode

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement run_episode()")


class Agent(Generic[T]):
    """
    Agent composition: Model + Harness.

    Attributes:
        model: The model used for reasoning/generation
        harness: The harness defining the rollout protocol
        name: Optional name for the agent
    """

    def __init__(
        self,
        model: Model[T],
        harness: AgentHarness[T],
        name: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            model: The model to use
            harness: The harness to use
            name: Optional agent name
        """
        self.model = model
        self.harness = harness
        self.name = name or f"Agent-{uuid.uuid4().hex[:8]}"
        logger.info(f"Initialized agent '{self.name}'")

    async def solve(
        self,
        task: Task[T],
        verifier: Optional[Verifier] = None,
        sandbox: Optional[Sandbox] = None,
    ) -> Trajectory:
        """
        Solve a single task.

        Args:
            task: The task to solve
            verifier: Optional verifier for scoring
            sandbox: Optional sandbox for isolation

        Returns:
            The trajectory of the episode
        """
        logger.info(f"Agent '{self.name}' solving task '{task.id}'")
        try:
            trajectory = await self.harness.run_episode(
                task=task,
                model=self.model,
                verifier=verifier,
                sandbox=sandbox,
            )
            logger.info(
                f"Agent '{self.name}' completed task '{task.id}' "
                f"with reward {trajectory.final_reward:.3f}"
            )
            return trajectory
        except Exception as e:
            logger.error(
                f"Agent '{self.name}' failed on task '{task.id}': {e}",
                exc_info=True,
            )
            raise


class Verifier(ABC):
    """
    Abstract base class for verifiers/scorers/graders.

    Scores an episode to calculate a reward, typically between 0 and 1.
    """

    @abstractmethod
    async def verify(
        self,
        task: Task,
        trajectory: Trajectory,
        state: State,
    ) -> float:
        """
        Verify and score an episode.

        Args:
            task: The task that was executed
            trajectory: The trajectory of the episode
            state: The final state of the episode

        Returns:
            A score between 0 and 1

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement verify()")


class Environment(Generic[T]):
    """
    Everything except the model.

    Environment = { Dataset, Agent Harness, Verifier, State }

    Attributes:
        dataset: The dataset of tasks
        harness: The agent harness
        verifier: The verifier for scoring
        sandbox: Optional sandbox for isolation
    """

    def __init__(
        self,
        dataset: Dataset[T],
        harness: AgentHarness[T],
        verifier: Optional[Verifier] = None,
        sandbox: Optional[Sandbox] = None,
    ):
        """
        Initialize the environment.

        Args:
            dataset: The dataset to use
            harness: The harness to use
            verifier: Optional verifier
            sandbox: Optional sandbox
        """
        self.dataset = dataset
        self.harness = harness
        self.verifier = verifier
        self.sandbox = sandbox
        logger.info(
            f"Initialized environment with dataset '{dataset.name}', "
            f"{len(dataset)} tasks"
        )

    async def run_task(
        self,
        task_index: int,
        model: Model[T],
    ) -> Trajectory:
        """
        Run a single task from the dataset.

        Args:
            task_index: Index of the task in the dataset
            model: The model to use

        Returns:
            The trajectory of the episode
        """
        task = self.dataset[task_index]
        agent = Agent(model=model, harness=self.harness)
        return await agent.solve(
            task=task,
            verifier=self.verifier,
            sandbox=self.sandbox,
        )

    async def run_all(
        self,
        model: Model[T],
        max_concurrent: int = 4,
    ) -> List[Trajectory]:
        """
        Run all tasks in the dataset.

        Args:
            model: The model to use
            max_concurrent: Maximum number of concurrent episodes

        Returns:
            List of trajectories for all tasks
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task_index: int) -> Trajectory:
            async with semaphore:
                return await self.run_task(task_index, model)

        tasks = [run_with_semaphore(i) for i in range(len(self.dataset))]
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Trajectory] = []
        for i, result in enumerate(trajectories):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                # Create a failed trajectory
                task = self.dataset[i]
                state = State(
                    episode_id=f"failed-{uuid.uuid4().hex[:8]}",
                    task=task,
                    status=TaskStatus.ERROR,