# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resources server for judge-graded search / BrowseComp-style QA rows.

Shape of the environment, and how it differs from resources_servers/swebench:

  seed_session  creates ONE general-purpose sandbox from a fixed image (no per-instance
                image derivation, no conda activation, no anti-cheat git scrub) and hands
                the agent back the handle pair it needs to re-attach.
  verify        has NO eval sandbox and runs no tests. It extracts the agent's final
                answer and grades it with an LLM judge, exactly as the reference
                simple-evals BrowseComp grader does.

This pairing lets the unmodified responses_api_agents/opencode_sandboxed_agent run a
search benchmark: the agent is told only the question, works inside the sandbox, and the
reward comes from the judge rather than from a test harness.
"""

import asyncio
import json
import re
import sys
from time import time
from typing import Any, ClassVar, Dict, Optional, Tuple

import aiohttp
from fastapi import Request, Response
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY

# The grader the apptainer comparators were scored with. NOT the longer
# swe_agents/judge_prompt.py on the legacy branch: those runs set
# use_search_judge=False and graded OFFLINE afterwards, so the in-run judge code never
# executed. Proof: every comparator row carries use_search_judge=False, reward=0.0 and no
# judge* keys at all. Presence in the legacy tree is not evidence of execution; check the
# run's instance_config before trusting a source file.
#
# CORRECTED. This comment used to claim the offline grader used by our comparison pipeline is
# BYTE-IDENTICAL to this template. That stopped being true once the stock simple-evals text
# here was replaced with the ordered-rules R1-R5 cascade; the two now differ in both length
# and content, and the comment simply went stale. Both paths call the same judge.
#
# What the drift is worth, measured by re-grading BOTH arms' answers under BOTH prompts against
# the same judge on the paired BrowseComp questions: the prompt choice moves each arm by well
# under a point, in opposite directions, and does NOT explain the gap between the two arms,
# which survives under either prompt at high significance. Quote that re-grade, not an identity
# claim, whenever these numbers are compared.
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE
from resources_servers.opencode_search.exa_cache import ExaCache


# A judge that echoes this instead of grading has repeated the template back at us.
_JUDGE_TEMPLATE_ECHO = "The final exact answer extracted from the [response]"


def _last_assistant_text(response: NeMoGymResponse) -> str:
    """Text of the LAST assistant message item in response.output (the model's final answer).

    Not Response.output_text, which concatenates every assistant message item — wrong when the
    model emits text alongside tool calls mid-trajectory. Skipping non-message items also keeps
    a trailing reasoning item (routine with uses_reasoning_parser) from masking the answer.
    """
    text = ""
    for item in response.output or []:
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            t = "".join(
                getattr(p, "text", "")
                for p in (getattr(item, "content", None) or [])
                if getattr(p, "type", None) == "output_text"
            )
            if t:
                text = t
    return text


def parse_judge_verdict(text: str) -> Tuple[bool, str, bool]:
    """Parse a judge reply into (is_correct, extracted_answer, parsed_ok).

    The LAST 'correct: yes|no' wins: reasoning judges echo the template inside their think
    block, so a first-match parse grades the echo instead of the verdict.
    """
    matches = list(re.finditer(r"correct:\s*(yes|no)\b", text or "", re.IGNORECASE))
    if not matches:
        return False, "", False

    is_correct = matches[-1].group(1).lower() == "yes"

    ans_matches = list(re.finditer(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text or ""))
    extracted = ans_matches[-1].group(1).strip() if ans_matches else ""

    if extracted and _JUDGE_TEMPLATE_ECHO in extracted:
        return False, "", False

    return is_correct, extracted, True


class SearchResourcesServerConfig(BaseResourcesServerConfig):
    # ---- Exa response cache -------------------------------------------------------------------
    # Shared by every shard and every sandbox: the store is a directory on Lustre, so it outlives
    # any job, and this server fronts it over HTTP because our agents are remote pods and cannot
    # reach a localhost sidecar the way the legacy per-node design did.
    #
    # Empty root disables it entirely and the proxy endpoint is not registered, so the cache is
    # opt-in per benchmark and an un-cached arm costs one config line.
    exa_cache_root: str = ""
    # The EXISTING ~2M-entry store, read-only, searched by exact key only. Not walked at startup:
    # building a fuzzy index over 2M Lustre files would hang the server while rollouts wait on
    # seed_session, whereas an exact lookup is a direct path from the hash and costs one stat().
    exa_cache_legacy_root: str = ""
    # Legacy entries are ~90% Tavily, holding structured results rather than rendered Exa text.
    # Serving them means translating (render_tavily). On by default: the model still gets real
    # search results and the point of the cache is cost.
    exa_cache_use_tavily: bool = True
    # 0.8 matches the legacy threshold. Measured by replaying real queries through this store,
    # fuzzy matching returns hits where exact-only returns none at all -- so this knob IS the
    # feature, not a tuning detail.
    exa_cache_fuzzy_threshold: float = 0.8
    # Upstream, overridable so a run can be pinned to a frozen mirror.
    exa_upstream_url: str = "https://mcp.exa.ai/mcp"

    # verify() needs only the saved response, question and ground truth — the sandbox is already
    # gone by then and the judge call is pure — so a saved run can be re-graded offline under a
    # different judge or prompt without re-running the model (gym eval reverify).
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Sandbox the agent works in. One fixed image for every row: these tasks carry no
    # per-instance repo, so there is nothing to derive an image from.
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_image: str

    # Grading. The judge is the only grader; there is no exact-match fallback, because a
    # BrowseComp answer is free text and string equality scores correct answers as wrong.
    judge_model_server: ModelServerRef
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    # Legacy ladder, copied field-for-field from the legacy stack's
    # swe_agents/app.py:151-168. 1 initial attempt + 3 retries, a FIXED 30s sleep between them
    # (that branch's comment records it replacing full-jitter exponential backoff on 09-03), and a
    # shorter cap on retries so a dead endpoint costs ~300 + 3*(30+120) = ~12.5 min per rollout
    # rather than the ~90 min our 15-attempt/300s-each schedule allowed. On a 4h wall where row
    # completeness is already the binding constraint, that difference is the whole budget.
    judge_max_attempts: int = 4
    judge_retry_delay_seconds: float = 30.0
    judge_retry_timeout_seconds: float = 120.0
    # Abort the run after this many rollouts in a row fail to grade. Guards the silent-starvation
    # mode where a dead judge masks every sample and the run burns its wall producing nothing.
    # Counted per ROLLOUT, not per attempt, so one transient burst cannot trip it. 0 disables.
    judge_max_consecutive_failures: int = 25
    # Budget covers reasoning + final message combined. 2048 makes reasoning judges burn the
    # whole budget thinking, emitting no final message, which reads as an unparseable verdict.
    judge_max_output_tokens: int = 10240
    # Bounds ONE attempt. Gym's shared aiohttp client has no timeout of its own, so without this
    # a judge that accepts the connection and never answers holds its rollout until the job's
    # wall clock and the retry loop below never runs.
    judge_timeout_seconds: float = 300.0


class SearchRunRequest(BaseRunRequest):
    # The agent forwards the whole dataset row, so tolerate columns we do not model.
    model_config = ConfigDict(extra="allow")

    question: str
    ground_truth: str


class SearchSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class SearchSeedSessionResponse(BaseSeedSessionResponse):
    # The only field the agent reads. Without it opencode_sandboxed_agent silently creates its
    # own sandbox from a hardcoded SWE-bench image instead of connecting to this one.
    sandbox_handle: str


class SearchVerifyRequest(SearchRunRequest, BaseVerifyRequest):
    pass


# How a rollout ended, for endings that are a BUDGET failure rather than a wrong answer. Same two
# agent_error_kinds the legacy stack flagged (the legacy stack's
# swe_agents/app.py:5742-5751, alongside eval/agent timeouts, OOM and judge_masked -- in this stack
# those three already land in the failures sidecar instead).
#
# SCOPE, and this is the part that is easy to get wrong: legacy's mask_sample gates the RL
# GRADIENT, not the eval metric. Its published comparator scored the capped rows: its denominator
# is the full row count including max_iteration endings. So we record the flag and score the row
# anyway; see aggregate_metrics. Excluding them would report a higher number against a bar that
# was computed the other way.
# agent_timed_out joins these: legacy masked it in the same branch (app.py:5742-5751, the
# eval_timed_out/agent_timed_out arm). Like the other two it is RECORDED, not excluded from
# the mean -- see aggregate_metrics.
MASKED_TERMINAL_REASONS = frozenset({"max_iteration", "context_window", "agent_timed_out"})


class SearchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_final_answer: str
    final_text: str
    judge_parsed: bool
    judge_attempts: int
    judge_time_taken: float
    # True when the rollout ran out of budget rather than answering wrongly. Recorded only: the row
    # keeps its real reward and is scored like any other, because that is how the comparator's
    # number was computed. Consumers that should skip these (RL gradient, error analysis) filter on
    # this field themselves.
    mask_sample: bool = False


class SearchResourcesServer(SimpleResourcesServer):
    config: SearchResourcesServerConfig

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    _session_id_to_sandbox: Dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    # Rollouts in a row whose judge never produced a verdict. Reset by any success.
    _judge_consecutive_failures: int = PrivateAttr(default=0)
    _exa_cache: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox = dict()

    async def _create_sandbox(self) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        sandbox_resources = SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {})))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        spec = SandboxSpec(
            image=self.config.sandbox_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to the image's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {"nemo_gym_agent": self.config.name},
            resources=sandbox_resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )

        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(spec)
        return sandbox

    async def seed_session(self, request: Request, body: SearchSeedSessionRequest) -> SearchSeedSessionResponse:
        sandbox = await self._create_sandbox()
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        return SearchSeedSessionResponse(sandbox_handle=sandbox._handle.sandbox_id)

    async def _call_judge(
        self,
        judge_create_params: NeMoGymResponseCreateParamsNonStreaming,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[Any, str]:
        judge_response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_create_params,
            response_model=NeMoGymResponse,
            timeout_seconds=(self.config.judge_timeout_seconds if timeout_seconds is None else timeout_seconds),
        )
        if not judge_response.output:
            raise JudgeError("empty judge response")
        return judge_response, _last_assistant_text(judge_response)

    async def _judge_answer(self, question: str, ground_truth: str, final_text: str) -> Dict[str, Any]:
        """Grade final_text against ground_truth. Returns a dict of result fields.

        Two failure modes are deliberately kept distinct: a judge that answered but could not be
        parsed is a legitimate zero, while a judge that never answered is infrastructure noise and
        raises JudgeError so judge_failsafe routes the row to the failures sidecar.
        """
        answer = re.sub(r"<think>.*?</think>", "", final_text or "", flags=re.DOTALL).strip()

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, correct_answer=ground_truth, response=answer
        )

        judge_create_params = (
            self.config.judge_responses_create_params.model_copy(deep=True)
            if self.config.judge_responses_create_params is not None
            else NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        judge_create_params.max_output_tokens = self.config.judge_max_output_tokens
        # Legacy graded at temperature 0.0; overriding to 1.0 here would silently undo the config
        # and make the grader itself a source of run-to-run variance.
        if self.config.judge_responses_create_params is None:
            judge_create_params.temperature = 0.0
        judge_create_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]

        started = time()
        last_error: Optional[str] = None
        # Did the judge ever actually reply? A judge that answers but cannot be parsed is a
        # legitimate zero and must stay in the scored population; only a judge we could never
        # reach is infrastructure noise worth routing to the failures sidecar. Without this the
        # two are conflated and a chatty-but-malformed judge silently shrinks the denominator.
        had_response = False
        for attempt in range(self.config.judge_max_attempts):
            try:
                # Attempt 1 gets the full timeout; retries get the shorter one, so the worst
                # case stays bounded instead of multiplying the full cap by the attempt count.
                timeout_s = (
                    self.config.judge_timeout_seconds if attempt == 0 else self.config.judge_retry_timeout_seconds
                )
                _judge_response, judge_text = await self._call_judge(judge_create_params, timeout_s)
                is_correct, extracted, parsed_ok = parse_judge_verdict(judge_text)
                if parsed_ok:
                    self._judge_consecutive_failures = 0
                    return {
                        "reward": 1.0 if is_correct else 0.0,
                        "extracted_final_answer": extracted,
                        "judge_parsed": True,
                        "judge_attempts": attempt + 1,
                        "judge_time_taken": time() - started,
                    }
                had_response = True
                last_error = f"unparseable verdict (tail={judge_text[-200:]!r})"
            except JudgeError as e:
                last_error = str(e)
            except Exception as e:  # noqa: BLE001 - any transport failure is a judge failure
                last_error = f"{type(e).__name__}: {e}"

            print(
                f"[judge_retry attempt={attempt + 1}/{self.config.judge_max_attempts} error={last_error}]",
                flush=True,
            )
            if attempt + 1 < self.config.judge_max_attempts:
                await asyncio.sleep(max(0.0, self.config.judge_retry_delay_seconds))

        if had_response:
            # The judge talked to us every time and we could never read a verdict out of it. That
            # is a graded outcome, not infrastructure noise: score it zero and keep the row in the
            # population, flagged so a regrade can revisit it.
            return {
                "reward": 0.0,
                "extracted_final_answer": "",
                "judge_parsed": False,
                "judge_attempts": self.config.judge_max_attempts,
                "judge_time_taken": time() - started,
            }

        # Never got a reply at all: the endpoint is down or unreachable. Raising sends the row to
        # the failures sidecar rather than recording a fabricated zero.
        self._judge_consecutive_failures += 1
        print(
            f"[search] judge unreachable after {self.config.judge_max_attempts} attempts; "
            f"last: {last_error} [{self._judge_consecutive_failures} consecutive]",
            flush=True,
        )
        self._maybe_abort_on_judge_failures()
        raise JudgeError(
            f"judge unreachable after {self.config.judge_max_attempts} attempts; last error: {last_error}"
        )

    def _maybe_abort_on_judge_failures(self) -> None:
        """Stop the run once the judge looks dead, rather than masking every remaining row.

        Ported from the legacy stack (the legacy stack's swe_agents/app.py).
        Without it a dead grader is indistinguishable from a model that answers nothing: every row
        routes to the failures sidecar, the headline metric is computed over whatever survived,
        and the run burns its whole wall clock producing a number from a handful of samples.
        """
        limit = self.config.judge_max_consecutive_failures
        if limit and self._judge_consecutive_failures >= limit:
            raise RuntimeError(
                f"judge appears dead: {self._judge_consecutive_failures} consecutive rollouts "
                f"failed to grade (judge_max_consecutive_failures={limit}). Aborting rather than "
                f"masking the rest of the run."
            )

    async def _release_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.pop(session_id, None)
        if sandbox is None:
            return
        # The agent also stops the sandbox after verify returns; both sides tolerate the other
        # having gone first. Releasing here keeps a crashed agent from leaking the pod.
        try:
            await sandbox.stop()
        except Exception as e:  # noqa: BLE001
            print(f"[search] sandbox stop failed (ignored): {type(e).__name__}: {e}", flush=True)

    # ---- Exa cache proxy ----------------------------------------------------------------------
    def setup_webserver(self):
        """Add the Exa proxy route alongside the standard resources-server endpoints.

        The plugin inside the sandbox POSTs its MCP JSON-RPC here instead of to mcp.exa.ai, so the
        ONLY plugin change is which base URL it uses. Everything else -- lookup, fuzzy matching,
        storage, provider rules, accounting -- lives here, on the Gym side, where it can be changed
        without touching a file that ships into the sandbox.
        """
        app = super().setup_webserver()
        if self.config.exa_cache_root:
            app.post("/mcp")(self._exa_proxy)
            app.get("/exa_cache_stats")(self._exa_cache_stats)
        return app

    @property
    def _cache(self) -> ExaCache:
        if self._exa_cache is None:
            self._exa_cache = ExaCache(
                self.config.exa_cache_root,
                self.config.exa_cache_fuzzy_threshold,
                legacy_root=self.config.exa_cache_legacy_root,
                use_tavily=self.config.exa_cache_use_tavily,
            )
        return self._exa_cache

    async def _exa_cache_stats(self) -> Dict[str, Any]:
        return dict(self._cache.stats)

    async def _exa_proxy(self, request: Request) -> Response:
        """Serve an Exa MCP call from cache, or forward it upstream and remember the answer.

        Speaks the endpoint's own wire format in both directions -- a single `data: ` SSE frame
        carrying the JSON-RPC envelope -- so the plugin cannot tell a hit from a miss apart from
        latency, and the contamination guard and TDM filter still run on the way out exactly as
        they do today (they are applied by the plugin AFTER this returns).

        Fails OPEN on every error path: a cache that is confused must cost money, never
        correctness. Any exception here forwards upstream unchanged.
        """
        body = await request.body()
        try:
            payload = json.loads(body)
            params = payload.get("params") or {}
            tool = params.get("name") or ""
            args = params.get("arguments") or {}
        except (json.JSONDecodeError, AttributeError):
            return await self._exa_forward(request, body)

        # search keys on the query, fetch keys on the url; they are separate sections of one store.
        if tool == "web_search_exa":
            kind, keytext = "search", str(args.get("query") or "")
        elif tool == "web_fetch_exa":
            urls = args.get("urls") or []
            kind, keytext = "browse", str(urls[0] if urls else "")
        else:
            return await self._exa_forward(request, body)

        # ExaCache.get/put are SYNCHRONOUS and touch Lustre. Calling them directly from this
        # async handler blocks the whole event loop -- not just this request -- because asyncio
        # is single-threaded. The first get() is the worst: it lazily builds the fuzzy index by
        # globbing "*/*/*.json" under the cache root and read_text()-ing every entry (2M entries
        # on the shared store), and every other in-flight search sits behind it. Measured on the
        # cached arm, a large share of searches died on the plugin's deadline, against virtually
        # none on the same harness with the cache off.
        #
        # asyncio.to_thread moves the blocking work to the default executor so the loop stays
        # free. It does NOT make the cache faster -- the first lookup still pays the index walk --
        # it stops that cost being charged to every concurrent rollout.
        if keytext:
            try:
                cached, how = await asyncio.to_thread(self._cache.get, kind, keytext)
            except Exception as e:  # noqa: BLE001 - a broken cache must not break a rollout
                print(f"[exa-cache] lookup failed ({type(e).__name__}): {e}", flush=True)
                cached, how = None, "error"
            if cached is not None:
                print(f"[exa-cache] HIT kind={kind} how={how} key={keytext[:70]!r}", flush=True)
                return Response(content=cached, media_type="text/event-stream")

        upstream = await self._exa_forward(request, body)
        if keytext and upstream.status_code == 200:
            try:
                # Also offloaded: put() writes a temp file and renames it on Lustre, where a
                # create has been measured at ~1% EIO under load and is never fast.
                await asyncio.to_thread(self._cache.put, kind, keytext, upstream.body.decode("utf-8", "replace"))
                print(f"[exa-cache] MISS kind={kind} stored key={keytext[:70]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[exa-cache] store failed ({type(e).__name__}): {e}", flush=True)
        return upstream

    async def _exa_forward(self, request: Request, body: bytes) -> Response:
        """Forward verbatim to Exa, preserving the query string so the per-row API key rides along."""
        qs = request.url.query
        url = self.config.exa_upstream_url + (f"?{qs}" if qs else "")
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    url,
                    data=body,
                    headers={"content-type": "application/json", "accept": "application/json, text/event-stream"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    raw = await resp.read()
                    return Response(
                        content=raw,
                        status_code=resp.status,
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[exa-cache] upstream failed ({type(e).__name__}): {e}", flush=True)
            return Response(content=b"", status_code=502)

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Report the headline mean over EVERY graded row, capped ones included.

        Deliberately does NOT drop mask_sample rows, even though the legacy stack set that flag for
        the same endings. mask_sample there gates what enters the RL GRADIENT, not what enters an
        eval metric: the published comparator's denominator is its full row count, and the
        ungraded-reason rows plus the agent_error_kind=max_iteration rows account for exactly that
        total -- so its capped rows were scored, not excluded. Dropping ours would shrink the
        denominator against a bar computed the other way: an accounting gift, not a result.

        The flag is still recorded per row, so a training consumer can mask on it and an analysis
        can split on it; only the headline number stays comparable.
        """
        masked = sum(1 for vr in body.verify_responses if vr.get("mask_sample"))
        if masked:
            print(
                f"[search] {masked}/{len(body.verify_responses)} row(s) carry mask_sample "
                f"(budget-exhausted); SCORED anyway to match the comparator's denominator",
                file=sys.stderr,
            )
        return await super().aggregate_metrics(body)

    async def verify(self, request: Request, body: SearchVerifyRequest) -> SearchVerifyResponse:
        await self._release_sandbox(request.session[SESSION_ID_KEY])

        # Forwarded by the agent from the turn-budget plugin's stderr line; "" on a normal ending.
        masked = getattr(body, "opencode_terminal_reason", "") in MASKED_TERMINAL_REASONS

        final_text = _last_assistant_text(body.response)
        if not final_text.strip():
            # The agent produced no answer at all. That is a real zero, not judge noise, and
            # spending a judge call on an empty string would only add latency.
            return SearchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_final_answer="",
                final_text="",
                judge_parsed=False,
                judge_attempts=0,
                judge_time_taken=0.0,
                mask_sample=masked,
            )

        result = await self._judge_answer(
            question=body.question,
            ground_truth=body.ground_truth,
            final_text=final_text,
        )

        return SearchVerifyResponse(
            **body.model_dump(),
            reward=result["reward"],
            extracted_final_answer=result["extracted_final_answer"],
            final_text=final_text,
            judge_parsed=result["judge_parsed"],
            judge_attempts=result["judge_attempts"],
            judge_time_taken=result["judge_time_taken"],
            mask_sample=masked,
        )


if __name__ == "__main__":
    SearchResourcesServer.run_webserver()
