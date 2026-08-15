from pydantic import BaseModel

from .constraints import PreferenceTier, ToolDefinition, ToolPreference


class ToolCallStep(BaseModel):
    tool: str
    args: dict
    reasoning: str | None = None   # why this tool was chosen — used by soft preference judge


class ToolCallTrace(BaseModel):
    steps: list[ToolCallStep]
    final_answer: str
    task_completed: bool


class IFToolPrefTrainingRecord(BaseModel):
    """A single training example for IF ToolPref RL.

    The preference is always stated in the system prompt, not restated per turn.
    The reference_trace shows the correct tool selection behavior.
    For SOFT tier, a contrastive_trace (preference violation) is also generated
    for DPO training alongside GRPO.
    """
    preference: ToolPreference
    primary_tool: ToolDefinition
    companion_tools: list[ToolDefinition]   # alternatives the model could plausibly choose
    system_prompt: str                      # embeds preference_instruction naturally
    user_query: str                         # ambiguous enough that both tools are plausible
    reference_trace: ToolCallTrace          # correct behavior: respects preference
    contrastive_trace: ToolCallTrace | None = None  # SOFT tier only: violates preference (for DPO)

    @property
    def all_available_tools(self) -> list[ToolDefinition]:
        return [self.primary_tool] + self.companion_tools
