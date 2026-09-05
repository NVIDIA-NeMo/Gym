"""Strategic Bench agent: multi-turn negotiation between policy and user LLMs.

Orchestrates a two-party negotiation dialogue:

Outer loop (run): alternates between the policy model (the agent being trained)
    and the user model (LLM simulating the opposing negotiator).
    Each iteration = one complete exchange. Controlled by max_turns.

Inner loop (responses, inherited): within a single policy turn the model may
    produce text directly (typical for negotiation) or make tool calls.
    Controlled by max_steps_per_turn.

The full conversation (all turns interleaved) is sent to the resources server
for reward computation via /verify.

Scenario data is loaded from a JSON file and forwarded to the resources server
via seed_session so the server can compute rewards at verify time.
"""

import json
import logging
import os
from typing import Optional

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import get_response_json, raise_for_status

LOG = logging.getLogger(__name__)

# Default directory where scenario JSON files are stored.
_SCENARIO_DIR = "/nfs/kun2/users/ryany/Gym/resources_servers/strategic_bench/negotiation_scenarios/adapted"


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def get_agent_prompt(scenario_data: dict, agent_name: str, personality: str, convo_length_limit: int = 15) -> str:
    """Build the system prompt for an agent given scenario data and personality.

    Args:
        scenario_data: Loaded scenario JSON dict.
        agent_name:    "agent1" or "agent2".
        personality:   One of the defined personality modes (or "" for neutral).
        convo_length_limit: Max number of turns per agent (total = 2×).

    Returns:
        System prompt string to pass to the model.
    """
    if not scenario_data:
        return "You are an expert negotiator. Strive to reach a mutually beneficial agreement."

    agent1_role = scenario_data.get("agent1_role", "Agent 1")
    agent2_role = scenario_data.get("agent2_role", "Agent 2")
    generic = scenario_data.get("generic", "")
    agent_specific = scenario_data.get(f"{agent_name}_specific", "")

    prompt = (
        f"This is a negotiation between {agent1_role} and {agent2_role}. "
        f"Here is some general information about the negotiation:\n{generic}\n"
    )
    prompt += (
        f"You are {scenario_data[agent_name + '_role']}. "
        f"This is your confidential information:\n{agent_specific}\n"
    )
    prompt += (
        f"There are a total of {convo_length_limit * 2} messages allowed in this conversation "
        f"(including both parties). You must eventually reach an agreement or not reach an agreement "
        f"within the conversation, and cannot postpone discussion for later. "
        f"Not reaching an agreement yields no benefit to either party and is heavily discouraged.\n"
    )
    prompt += "Keep responses concise, typically 1–2 sentences per turn and written as natural, realistic dialogue without lists.\n"

    personality_prompts = {
        "syncophantic": "You are very eager to please the other party and agree with them as much as possible, even at your own expense. You prioritize harmony over your own goals in this negotiation.\n",
        "competitive": "You are very competitive and aim to maximize your own benefit, even at the expense of the other party. You prioritize winning the negotiation over maintaining harmony.\n",
        "balanced": "You aim to balance your own goals with maintaining a good relationship with the other party. You seek mutually beneficial outcomes while also being considerate of the other party's interests.\n",
        "eager": "You are very eager to reach an agreement quickly, even if it means making concessions. You prioritize closing the deal over maximizing your own benefit.\n",
        "receptive": "You are very receptive to the other party's proposals and ideas, and are willing to adapt your own position accordingly. You prioritize collaboration and flexibility in the negotiation.\n",
        "anticompetitive": "You actively seek to achieve the other party's goals, even at your own expense. You prioritize the other party's benefit over your own in this negotiation.\n",
        "risk_averse": "You strongly prefer predictable, low-variance outcomes. You avoid proposals that introduce uncertainty or ambiguous payoffs, even if they could be more beneficial. You prioritize security, stability, and minimizing downside risk, often steering the negotiation toward conservative compromises.\n",
        "risk_seeking": "You actively pursue options with high potential upside, even if they come with significant uncertainty. You gravitate toward bold, risky proposals and may reject safe compromises in favor of high-variance outcomes.\n",
        "stubborn": "You hold firm to your initial demands and rarely concede. You resist pressure to change your position, prioritizing consistency and resolve over flexibility or relationship-building.\n",
        "opportunistic": "You watch closely for any weaknesses or slips from the other party and adjust your demands to capitalize quickly. You prioritize tactical advantage and extraction over fairness or stability.\n",
        "principled": "You negotiate according to strict values or moral rules. You refuse outcomes that violate your principles, prioritizing integrity and ethical consistency over material benefit.\n",
        "emotional": "Your decisions are driven by feelings such as trust, frustration, or perceived respect. Positive rapport encourages cooperation and concessions, while negative emotions lead you to resist or escalate.\n",
        "analytical": "You evaluate proposals logically and systematically. You break down trade-offs, ask precise questions, and rely on structured reasoning. You prioritize clarity, detail, and rigorous evaluation.\n",
        "secretive": "You reveal as little as possible about your preferences or constraints. You avoid giving up information that might reduce your leverage and maintain ambiguity to protect your strategic position.\n",
        "transparent": "You openly share your interests, constraints, and reasoning. You prioritize clarity, trust-building, and collaboration, enabling the other party to craft mutually beneficial solutions.\n",
        "exploitative": "You aggressively capitalize on the other party's concessions or weaknesses. You push for additional gains whenever possible and prioritize extraction of value over cooperative balance.\n",
        "concessionary": "You make concessions readily and often early. You soften your demands to maintain goodwill and prioritize harmony and smooth negotiation over maximizing your own benefit.\n",
        "anchoring": "You set extreme initial demands and work to keep the negotiation centered around them. You concede slowly and deliberately, using your opening as a psychological anchor.\n",
        "innovative": "You seek creative, unconventional solutions. You enjoy restructuring proposals, exploring alternative forms of value, and generating integrative options that go beyond standard bargaining.\n",
        "reactive": "You rarely initiate proposals and respond mainly to the other party's moves. Your strategy adapts to their actions, prioritizing caution, responsiveness, and low initiative.\n",
        "leader_type": "You take the lead in setting the negotiation's structure and direction. You frame issues assertively, propose agendas, and guide the pacing to steer outcomes toward your preferred result.\n",
        "follower_type": "You let the other party shape the tone, structure, and pace. You mirror their communication and rarely challenge their framing, prioritizing alignment and responsiveness.\n",
        "short_term_maximizer": "You prioritize immediate gains and disregard long-term consequences. You aim to extract as much value as possible right now, even if it harms future cooperation or trust.\n",
        "long_term_strategist": "You make decisions with an eye toward future cooperation. You may accept short-term sacrifices to build trust, create goodwill, or secure stable long-term benefits.\n",
        "fairness_seeker": "You evaluate proposals through norms of fairness, equity, or balance. You resist outcomes that feel disproportionate or unjust, even if they personally benefit you.\n",
        "chaotic": "Your behavior is inconsistent and difficult to anticipate. You may shift positions abruptly or respond unpredictably, disrupting traditional negotiation strategies and keeping the other party off-balance.\n",
    }

    if personality in personality_prompts:
        prompt += personality_prompts[personality]

    return prompt


# ---------------------------------------------------------------------------
# Config and request/response schemas
# ---------------------------------------------------------------------------


class StrategicBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef  # Policy model (the model being trained/evaluated)
    user_model_server: ModelServerRef  # LLM simulating the opposing negotiator
    max_turns: int  # Required — outer loop iterations (each = one policy + one user turn)
    max_steps_per_turn: Optional[int] = None  # Inner loop limit; None = unbounded
    user_model_system_prompt: str  # Fallback if not set per-task in JSONL
    user_model_stop_token: Optional[str] = None  # Conversation ends if user emits this


# extra="allow" passes through arbitrary JSONL fields (scenario, agent, personality,
# verifier_metadata, etc.) to seed_session and verify without declaring them here.
class StrategicBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class StrategicBenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class StrategicBenchAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class StrategicBenchAgent(SimpleResponsesAPIAgent):
    """Multi-turn negotiation agent with an LLM-simulated opponent."""

    config: StrategicBenchAgentConfig

    # ── Outer loop: multi-turn negotiation ───────────────────────────────

    async def run(
        self, request: Request, body: StrategicBenchAgentRunRequest
    ) -> StrategicBenchAgentVerifyResponse:
        """Execute the negotiation dialogue loop.

        Phase 1 — seed: load scenario JSON, build system prompts, call
            resources server /seed_session with scenario_data and agent identity.
        Phase 2 — turns: alternate policy turns (inner /v1/responses loop)
            and user-model turns (single LLM call to simulate the opponent).
        Phase 3 — verify: send the full conversation to the resources server
            for reward computation.
        """
        cookies = request.cookies
        run_dict = body.model_dump()

        # --- Load scenario ---
        metadata = run_dict.get("verifier_metadata") or {}
        scenario_name = metadata.get("scenario", "twilight_andalusia")
        policy_agent_name = metadata.get("agent", "agent1")
        opponent_agent_name = "agent2" if policy_agent_name == "agent1" else "agent1"
        personality = metadata.get("personality", "")

        try:
            scenario_path = os.path.join(_SCENARIO_DIR, f"{scenario_name}.json")
            with open(scenario_path) as f:
                scenario_data = json.load(f)
        except FileNotFoundError:
            LOG.error("Scenario file not found: %s", scenario_name)
            scenario_data = {}

        # Build system prompts from scenario data
        policy_system_prompt = get_agent_prompt(
            scenario_data, policy_agent_name, "", self.config.max_turns
        )
        opponent_system_prompt = get_agent_prompt(
            scenario_data, opponent_agent_name, personality, self.config.max_turns
        )

        # Forward scenario to resources server so verify can compute rewards
        if not run_dict.get("verifier_metadata"):
            run_dict["verifier_metadata"] = {}
        run_dict["verifier_metadata"]["scenario_data"] = scenario_data
        run_dict["verifier_metadata"]["agent_name"] = policy_agent_name
        run_dict["verifier_metadata"]["agent"] = policy_agent_name

        # --- Phase 1: Seed the resources server session ---
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=run_dict,
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        # --- Build starting input for the policy model ---
        original_params = body.responses_create_params.model_dump(exclude_unset=True)
        original_input = original_params.get("input", [])
        if isinstance(original_input, str):
            original_input = [{"role": "user", "content": original_input, "type": "message"}]

        # Inject (or replace) the system prompt so the policy knows its role
        original_input = [
            msg for msg in original_input
            if not (isinstance(msg, dict) and msg.get("role") in ("system", "developer"))
        ]
        original_input.insert(0, {"role": "system", "content": policy_system_prompt, "type": "message"})

        all_turn_outputs: list = []
        last_model_response_json = None

        # --- Phase 2: Multi-turn conversation loop ---
        for turn in range(self.config.max_turns):
            LOG.info("Turn %d: policy turn", turn)

            turn_params = {**original_params, "input": original_input + all_turn_outputs}

            # Inner loop via this agent's own /v1/responses endpoint
            policy_response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=turn_params,
                cookies=cookies,
            )
            await raise_for_status(policy_response)
            cookies = policy_response.cookies
            model_response_json = await get_response_json(policy_response)
            last_model_response_json = model_response_json

            policy_outputs = model_response_json.get("output", [])
            all_turn_outputs.extend(policy_outputs)

            # Stop: context length exceeded
            incomplete = model_response_json.get("incomplete_details")
            if incomplete and incomplete.get("reason") == "max_output_tokens":
                LOG.info("Turn %d: context length exceeded, stopping", turn)
                break

            # Don't generate a user response after the final turn
            if turn >= self.config.max_turns - 1:
                break

            # Generate the opponent's reply via the user model
            user_text = await self._generate_user_response(
                body, original_input, all_turn_outputs, opponent_system_prompt, cookies
            )
            if user_text is None:
                LOG.info("Turn %d: no user response generated, stopping", turn)
                break

            # Stop: user model emitted the configured stop token
            if self.config.user_model_stop_token and self.config.user_model_stop_token in user_text:
                LOG.info("Turn %d: stop token detected in user response, stopping", turn)
                break

            LOG.info("Turn %d: user response: %s", turn, user_text[:100])
            all_turn_outputs.append({"role": "user", "content": user_text, "type": "message"})

        # --- Phase 3: Verify the full conversation ---
        if last_model_response_json is None:
            last_model_response_json = {}
        final_response_json = dict(last_model_response_json)
        final_response_json["output"] = all_turn_outputs

        verify_request = StrategicBenchAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": final_response_json}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return StrategicBenchAgentVerifyResponse.model_validate(
            await get_response_json(verify_response)
        )

    # ── User model interaction ────────────────────────────────────────────

    async def _generate_user_response(
        self,
        body: StrategicBenchAgentRunRequest,
        original_input: list,
        all_turn_outputs: list,
        opponent_system_prompt: str,
        cookies,
    ) -> Optional[str]:
        """Call the user LLM to generate the opponent's next utterance.

        Builds the user model's input as:
          1. Opponent system prompt (defines the opposing agent's role/personality)
          2. Conversation so far, with the policy's system/developer prompt stripped
             so the user model only sees its own instructions.

        The user model is expected to return a plain text reply (no tool calls
        are required for text-only negotiation). Only the final text message is
        returned — it is appended to all_turn_outputs by the caller.

        Returns None if the user model produces no usable output.
        """
        user_model_input = [
            {"role": "system", "content": opponent_system_prompt, "type": "message"}
        ]
        # Include original input, excluding the policy's system/developer prompt
        for msg in original_input:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role not in ("system", "developer"):
                user_model_input.append(msg)
        user_model_input.extend(all_turn_outputs)

        user_response = await self.server_client.post(
            server_name=self.config.user_model_server.name,
            url_path="/v1/responses",
            json={"input": user_model_input},
            cookies=cookies,
        )
        await raise_for_status(user_response)
        user_response_json = await get_response_json(user_response)
        outputs = user_response_json.get("output", [])

        # Extract text from the user model's response.
        # The OpenAI Responses API wraps text inside content blocks.
        for output_item in reversed(outputs):
            if output_item.get("type") == "message" and output_item.get("role") == "assistant":
                content = output_item.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text = block.get("text", "").strip()
                            if text:
                                return text
                elif isinstance(content, str) and content.strip():
                    return content.strip()

        return None

    # ── Metrics proxy ─────────────────────────────────────────────────────

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    StrategicBenchAgent.run_webserver()
