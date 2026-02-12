from pathlib import Path
from typing import Any, Literal

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


class Terminus2NemoGym(Terminus2):
    """Terminus2 variant that uses a NeMo Gym model server-compatible BaseLLM."""

    @staticmethod
    def name() -> str:
        return "terminus-2-nemo-gym"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "default"]
        | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        trajectory_config: dict | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        llm: BaseLLM | None = None,
        interleaved_thinking: bool = False,
        nemo_model_server_api_key: str = "placeholder",
        nemo_model_server_timeout_sec: float = 120.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if llm is None:
            if model_name is None:
                raise ValueError("model_name is required for Terminus2NemoGym")
            if api_base is None:
                raise ValueError(
                    "api_base is required for Terminus2NemoGym when llm is not provided"
                )

            llm = NemoGymLLM(
                model_name=model_name,
                api_base=api_base,
                api_key=nemo_model_server_api_key,
                temperature=temperature,
                collect_rollout_details=collect_rollout_details,
                reasoning_effort=reasoning_effort,
                model_info=model_info,
                timeout_sec=nemo_model_server_timeout_sec,
            )

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            enable_summarize=enable_summarize,
            proactive_summarization_threshold=proactive_summarization_threshold,
            max_thinking_tokens=max_thinking_tokens,
            model_info=model_info,
            trajectory_config=trajectory_config,
            tmux_pane_width=tmux_pane_width,
            tmux_pane_height=tmux_pane_height,
            store_all_messages=store_all_messages,
            record_terminal_session=record_terminal_session,
            llm=llm,
            interleaved_thinking=interleaved_thinking,
            *args,
            **kwargs,
        )
