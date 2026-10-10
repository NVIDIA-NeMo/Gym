"""Strategic Bench resources server: multi-turn negotiation environment.

The policy model plays as one agent (e.g. agent1) and the user model plays
as the opposing agent (e.g. agent2). Both produce plain text utterances —
no tool calls are used. The server tracks the session state and computes
rewards at the end of the conversation.

Session lifecycle:
  seed_session  → loads scenario data and agent identities from verifier_metadata
  (conversation turns managed by the agent, not the resources server)
  verify        → extracts full conversation from response output, detects
                  agreement, evaluates reward functions, returns reward
"""

import re
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State and schema models
# ---------------------------------------------------------------------------


class GameState(BaseModel):
    """Per-session negotiation state stored on the resources server."""

    scenario_data: Dict[str, Any] = Field(default_factory=dict)
    agent_name: str = "agent1"  # Which role the policy model plays, can be "agent1" or "agent2"
    game_over: bool = False
    turns: int = 0


class StrategicBenchConfig(BaseResourcesServerConfig):
    max_turns: int = 15 # Max turns before automatic termination with no agreement
    # Eval LLM used for agreement detection and reward function evaluation.
    # Populated from env.yaml via ${policy_base_url}, ${policy_api_key},
    # ${policy_model_name} — the same OpenAI-compatible endpoint as the policy.
    eval_model_base_url: str
    eval_model_name: str
    eval_model_api_key: str


class StrategicBenchSeedSessionRequest(BaseSeedSessionRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class MakeOfferRequest(BaseModel):
    offer_text: str
    speaker: Optional[str] = None  # "agent1" or "agent2"; inferred from turn order if None


class MakeOfferResponse(BaseModel):
    success: bool
    game_over: bool
    winner: Optional[str]
    message: str


class StrategicBenchVerifyRequest(BaseVerifyRequest):
    pass


class StrategicBenchVerifyResponse(BaseVerifyResponse):
    game_result: Optional[str] = None


# ---------------------------------------------------------------------------
# Server implementation
# ---------------------------------------------------------------------------


class StrategicBenchServer(SimpleResourcesServer):
    config: StrategicBenchConfig
    session_id_to_game: Dict[str, GameState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/make_offer")(self.make_offer)
        return app

    # ── Session initialisation ────────────────────────────────────────────

    async def seed_session(
        self, request: Request, body: StrategicBenchSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        """Initialise a negotiation session.

        verifier_metadata fields consumed:
          scenario_data  – full scenario JSON dict (loaded by the agent)
          agent          – "agent1" or "agent2" (which role the policy plays)
        """
        session_id = request.session[SESSION_ID_KEY]
        metadata = body.verifier_metadata or {}

        scenario_data = metadata.get("scenario_data", {})
        agent_name = metadata.get("agent", metadata.get("agent_name", "agent1"))

        game = GameState(scenario_data=scenario_data, agent_name=agent_name)
        self.session_id_to_game[session_id] = game
        return BaseSeedSessionResponse()

    # ── Optional offer tracking ───────────────────────────────────────────

    async def make_offer(self, request: Request, body: MakeOfferRequest) -> MakeOfferResponse:
        """Record a single negotiation utterance.

        This endpoint is optional in the text-only flow — the full conversation
        is also available via body.response.output in verify. It exists so that
        future tool-call-based variants can use it directly.
        """
        session_id = request.session[SESSION_ID_KEY]
        game = self.session_id_to_game.get(session_id)
        if game is None:
            return MakeOfferResponse(
                success=False,
                game_over=False,
                winner=None,
                message="No active session. Call seed_session first.",
            )

        if game.game_over:
            return MakeOfferResponse(
                success=False,
                game_over=True,
                winner=None,
                message="Negotiation is already over.",
            )

        game.turns += 1

        # Check for hard termination via stop token or max turns.
        done_token = "[DONE]"
        if done_token in body.offer_text or game.turns >= self.config.max_turns * 2:
            game.game_over = True

        return MakeOfferResponse(
            success=True,
            game_over=game.game_over,
            winner=None,
            message=f"Offer recorded (turn {game.turns}).",
        )

    # ── Conversation extraction ───────────────────────────────────────────

    def _extract_conversation(self, outputs: list) -> str:
        """Build a plain-text conversation string from the response output list.

        Handles two message formats produced by the NeMo Gym agent:
          - Policy turns: {"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "..."}]}
          - User-model turns: {"type": "message", "role": "user",
                               "content": "<plain string>"}
        """
        parts = []
        for item in outputs:
            # Support both dicts (JSON-decoded) and Pydantic objects
            if isinstance(item, dict):
                item_type = item.get("type")
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                item_type = getattr(item, "type", None)
                role = getattr(item, "role", "")
                content = getattr(item, "content", "")

            if item_type != "message":
                continue

            if isinstance(content, list):
                # Policy model format: list of content blocks
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "output_text":
                            text = block.get("text", "").strip()
                            if text:
                                parts.append(f"{role}: {text}")
                            break
                    else:
                        if getattr(block, "type", None) == "output_text":
                            text = getattr(block, "text", "").strip()
                            if text:
                                parts.append(f"{role}: {text}")
                            break
            elif isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content.strip()}")

        return "\n".join(parts)

    # ── Agreement detection ───────────────────────────────────────────────

    def _check_agreement_llm(self, conversation: str, scenario_data: dict) -> bool:
        """Use the eval LLM to determine whether both agents reached an agreement.

        Mirrors the agreement_summary + terminate_conversation logic from
        negotiation_environment_hrl.py: asks the eval model independently from
        each agent's perspective whether an agreement was reached. Both must
        answer YES for agreement=True.

        The prompt for each agent includes the scenario context (generic
        description + agent-specific confidential information) so the LLM can
        judge whether the terms discussed are actually acceptable to that party.

        Returns False (→ baseline reward) if the API call fails, so a transient
        eval-model error does not crash the episode.
        """
        llm = self._make_llm_callable()

        agent1_role = scenario_data.get("agent1_role", "Agent 1")
        agent2_role = scenario_data.get("agent2_role", "Agent 2")
        generic = scenario_data.get("generic", "")
        agent1_specific = scenario_data.get("agent1_specific", "")
        agent2_specific = scenario_data.get("agent2_specific", "")

        def is_yes(response: str) -> bool:
            return bool(response) and response.strip().lower().startswith("yes")

        def ask_agent(role: str, other_role: str, specific: str) -> bool:
            """Ask the LLM, acting as `role`, whether it agreed with `other_role`."""
            prompt = (
                f"{generic}\n"
                f"You are {role}. {specific}\n\n"
                f"Conversation:\n{conversation}\n\n"
                f"Given the conversation above, as {role}, have you concluded and "
                f"reached an agreement with {other_role}? "
                f"Respond with exactly one word: YES or NO."
            )
            return is_yes(llm(prompt))

        agent1_agreed = ask_agent(agent1_role, agent2_role, agent1_specific)
        if not agent1_agreed:
            LOG.info("_check_agreement_llm: agent1 (%s) did not agree", agent1_role)
            return False
        agent2_agreed = ask_agent(agent2_role, agent1_role, agent2_specific)
        LOG.info(
            "_check_agreement_llm: agent1=%s agreed=%s, agent2=%s agreed=%s",
            agent1_role, agent1_agreed, agent2_role, agent2_agreed,
        )
        return agent2_agreed

    # ── Reward calculation ────────────────────────────────────────────────

    def _make_llm_callable(self) -> Callable[[str], str]:
        """Return a synchronous LLM callable for reward functions and agreement checks.

        Uses the OpenAI-compatible endpoint configured via eval_model_base_url,
        eval_model_api_key, and eval_model_name — sourced from env.yaml at
        startup (${policy_base_url}, ${policy_api_key}, ${policy_model_name}).

        Returns "" on any API error so that a failed call scores 0 rather than
        crashing the episode.
        """
        base_url = self.config.eval_model_base_url
        model = self.config.eval_model_name
        api_key = self.config.eval_model_api_key

        def llm(prompt: str) -> str:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=64,
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                LOG.error("Eval LLM error: %s", e)
                return ""

        return llm

    def calculate_rewards(self, conversation: str, scenario_data: dict, agent_name: str) -> float:
        """Evaluate the scenario's reward functions against the conversation.

        Reward functions are stored as eval-able Python expression strings in
        scenario_data["{agent_name}_rewards"]. Each expression is evaluated
        with the following names in scope:
          CONTEXT  – plain-text conversation string
          llm      – callable that queries the eval model (may return "" if not
                     configured, causing LLM-dependent sub-rewards to score 0)
          re       – Python re module

        Returns the mean of all reward components, or 0.05 if no functions are
        defined (small baseline reward for completing the episode).
        """
        reward_functions = scenario_data.get(f"{agent_name}_rewards", [])

        llm = self._make_llm_callable()
        rewards = []

        for fn_str in reward_functions:
            try:
                result = eval(fn_str, {"CONTEXT": conversation, "llm": llm, "re": re}, {})
                if isinstance(result, list):
                    # Some reward functions return a list; take the mean of
                    # non-None numeric values.
                    numeric = [float(r) for r in result if r is not None]
                    result = float(np.mean(numeric)) if numeric else 0.0
                rewards.append(float(result))
            except Exception as e:
                LOG.error("Error evaluating reward function: %s", e)
                rewards.append(0.0)

        return float(np.mean(rewards)) if rewards else 0.05

    # ── Verification ─────────────────────────────────────────────────────

    async def verify(
        self, request: Request, body: StrategicBenchVerifyRequest
    ) -> StrategicBenchVerifyResponse:
        """Compute the reward for a completed negotiation episode.

        Steps:
          1. Retrieve the session state (scenario and agent identity).
          2. Extract the full conversation from body.response.output.
          3. Detect whether an agreement was reached.
          4. If agreed: evaluate scenario reward functions → reward.
             If not:    return a small baseline reward (0.05).
        """
        session_id = request.session[SESSION_ID_KEY]
        game = self.session_id_to_game.get(session_id)

        if game is None:
            return StrategicBenchVerifyResponse(
                **body.model_dump(), reward=0.0, game_result="no_game"
            )

        # --- Extract conversation from response output ---
        response_field = getattr(body, "response", None)
        if response_field is None:
            outputs = []
        elif isinstance(response_field, dict):
            outputs = response_field.get("output", [])
        else:
            outputs = getattr(response_field, "output", [])

        conversation = self._extract_conversation(outputs)

        # --- Determine outcome: ask each agent's perspective via eval LLM ---
        agreed = self._check_agreement_llm(conversation, game.scenario_data)
        LOG.info("verify: agreement=%s, conversation_length=%d chars", agreed, len(conversation))

        if agreed:
            reward = self.calculate_rewards(conversation, game.scenario_data, game.agent_name)
            game_result = "agreement"
        else:
            reward = 0.0
            game_result = "no_agreement"

        return StrategicBenchVerifyResponse(
            **body.model_dump(), reward=reward, game_result=game_result
        )


if __name__ == "__main__":
    StrategicBenchServer.run_webserver()
