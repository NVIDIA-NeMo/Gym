from enum import Enum
from typing import Literal
from pydantic import BaseModel

from .constraints import AgenticConstraint, ConversationalConstraint, ConstraintScope


class EpisodeType(str, Enum):
    AGENTIC = "agentic"           # 10–50+ ReAct steps; constraints govern reasoning/actions
    CONVERSATIONAL = "conversational"  # 3–8 dialogue turns; constraints govern text responses


class StepType(str, Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ConstraintPlacement(str, Enum):
    """Where in the system prompt the constraint appears.

    Placement must vary across training examples to cover the depth-decay failure mode
    (model follows a constraint stated early but forgets it as the system prompt grows).
    """
    PROMINENT = "prominent"   # top of system prompt, clearly labelled
    BURIED = "buried"         # after 2,000–80,000 chars of other instructions
    INLINE = "inline"         # delivered mid-task via a tool output or user turn


class TrajectoryStep(BaseModel):
    step_type: StepType
    content: str
    tool_name: str | None = None


class ReferenceTrajectory(BaseModel):
    steps: list[TrajectoryStep]
    task_completed: bool


class IFFormatTrainingRecord(BaseModel):
    """A single training example for IF Format RL.

    The constraint_set defines what must be verified at each applicable step.
    The system_prompt embeds those constraints. The reference_trajectory is used
    only for verifier calibration and SFT warmup — RL generates its own rollouts
    from (system_prompt, task_description) at training time.
    """
    episode_type: EpisodeType
    task_domain: str                          # e.g. "code_fix", "web_nav", "research"
    task_description: str
    completion_criterion: str                 # what "task completed" means for this task
    constraint_placement: ConstraintPlacement
    agentic_constraints: list[AgenticConstraint] = []
    conversational_constraints: list[ConversationalConstraint] = []
    system_prompt: str
    reference_trajectory: ReferenceTrajectory | None = None  # None means RL-only, no SFT use

    def active_constraints_for_step(self, step_type: StepType) -> list[AgenticConstraint]:
        """Constraints applicable to a given step type, for use by verifiers.

        Note: AFTER_TOOL_CALL constraints govern the assistant text FOLLOWING an
        observation; the canonical scope filter lives in
        resources_servers/if_agentic/app.py (_matches_scope) — keep in sync.
        """
        scope_map: dict[StepType, set[ConstraintScope]] = {
            StepType.THINKING: {ConstraintScope.ALL_STEPS, ConstraintScope.REASONING_STEPS},
            StepType.TOOL_CALL: {ConstraintScope.ALL_STEPS, ConstraintScope.CODE_STEPS},
            StepType.OBSERVATION: {ConstraintScope.ALL_STEPS, ConstraintScope.AFTER_TOOL_CALL},
            StepType.FINAL_ANSWER: {ConstraintScope.ALL_STEPS, ConstraintScope.FINAL_OUTPUT},
        }
        applicable_scopes = scope_map.get(step_type, set())
        return [c for c in self.agentic_constraints if c.scope in applicable_scopes]
