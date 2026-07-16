# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import orjson
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)

from collections import defaultdict

from omegaconf import DictConfig
from nemo_gym.global_config import get_wandb_run
from wandb import Table
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.path_utils import resolve_input_path
NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"


def _rollout_verify_debug_summary(row: Dict[str, Any], resources_server_name: str) -> Dict[str, Any]:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": agent_ref.get("name") if isinstance(agent_ref, dict) else None,
        "resources_server_name": resources_server_name,
    }
    return {k: v for k, v in summary.items() if v is not None}


# ---------------------------------------------------------------------------
# Agent-name → resources-server-name routing helpers
# Used by RolloutReverificationHelper to resolve which resources server to call
# for each rollout row, given a Hydra global config dict.
# ---------------------------------------------------------------------------


def _agent_to_rs_mapping_from_agent_blocks(global_conflict_dict: Union[Dict[str, Any], "DictConfig"]) -> Dict[str, str]:
    mapping = {}
    for name in global_conflict_dict:
        block = global_conflict_dict[name]
        if isinstance(block, (dict, DictConfig)) and "responses_api_agents" in block:
            impl = next(iter(block["responses_api_agents"].values()))
            rs = (impl.get("resources_server") or {}).get("name")
            if rs:
                mapping[name] = rs
    return mapping


def _agent_to_rs_mapping_from_resources_only_config(global_conflict_dict: Union[Dict[str, Any], "DictConfig"]) -> Dict[str, str]:
    # The rollout rows still carry agent names that were never started, so fall back to the
    # single resources server for EVERY requested key.
    resources_server_names = [
        name for name in global_conflict_dict
        if isinstance(global_conflict_dict[name], (dict, DictConfig)) and "resources_servers" in global_conflict_dict[name]
    ]
    if len(resources_server_names) == 1:
        only = resources_server_names[0]
        return defaultdict(lambda: only)  # any key → the one resources server instance
    if not resources_server_names:
        raise ConfigError("reverify: no resources server found in the config.")
    raise ConfigError(
        f"reverify: multiple resources servers {resources_server_names} and no agent blocks to "
        "route by. Use a config with agent blocks."
    )


def _build_agent_to_resources_server_mapping(global_conflict_dict: Union[Dict[str, Any], "DictConfig"]) -> Dict[str, str]:
    mapping = _agent_to_rs_mapping_from_agent_blocks(global_conflict_dict)
    if mapping:
        return mapping
    return _agent_to_rs_mapping_from_resources_only_config(global_conflict_dict)
    
class RolloutReverificationConfig(BaseNeMoGymCLIConfig):
    # to do - we provide description 2 times here - once in the config main.py and once in the field
    # to do can we use Path already here?
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by `gym eval run`."
    )
    rollouts_jsonl_fpath: str = Field(
        description="The file path of the rollouts to re-verify, as output by `gym eval run`."
    )
    output_jsonl_fpath: str = Field(description="The output data jsonl file path with recomputed rewards.")
    force: bool = Field(
        default=False,
        description=(
            "Re-verify even against servers whose reverify_mode is UNSUPPORTED (rewards may be "
            "incorrect); output filenames are prefixed with `unsafe_`."
        ),
    )
    judge_failed_only: bool = Field(
        default=False,
        description=(
            "Only re-verify rollouts whose judge call previously failed; successful rows are copied through unchanged."
        ),
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-reverification aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    num_samples_in_parallel: Optional[int] = Field(
        default=None, description="Maximum number of samples to re-verify in parallel."
    )
    limit: Optional[int] = Field(default=None, description="Maximum number of examples to re-verify.")
    upload_rollouts_to_wandb: bool = Field(
        default=True,
        description="Upload the rollouts to W&B. Sometimes this should be off because the rollouts are massive. Default: True",
    )



class RolloutReverificationHelper(BaseModel):
    def setup_server_client(self, head_server_config: Optional[BaseServerConfig] = None) -> "ServerClient":  # pragma: no cover
        server_client = ServerClient.load_from_global_config(head_server_config)
        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(
                cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict)
            )
        return server_client

    def _yield_inputs_and_rollouts_paired(
        self, materialized_inputs_jsonl_fpath: Path, rollouts_jsonl_fpath: Path, limit: Optional[int] = None
    ) -> Iterator[Tuple[Dict, Dict]]:
        inputs_by_key = {}
        with open(materialized_inputs_jsonl_fpath) as m_f:
            for line in m_f:
                r = orjson.loads(line)
                inputs_by_key[(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])] = r
        # to do - change to use namedtuple
        # to do - tqdm or other progress bar already used in Gym
        with open(rollouts_jsonl_fpath) as r_f:
            for i, line in enumerate(r_f):  # never holds the whole file
                if limit and i >= limit:
                    break
                rollout_row = orjson.loads(line)
                input_row = inputs_by_key.get((rollout_row[TASK_INDEX_KEY_NAME], rollout_row[ROLLOUT_INDEX_KEY_NAME]))
                if input_row is None:
                    raise ConfigError(f"No matching materialized input row found for rollout row {rollout_row}")
                yield (input_row, rollout_row)

    def _run_reverification_payloads(
        self,
        payloads: List[Dict],
        semaphore: Semaphore | nullcontext[None] | None = None,
    ) -> Iterator[Future]:  # pragma: no cover
       
        semaphore = semaphore or nullcontext[None]()
        server_client = self.setup_server_client()
        agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            async with semaphore:
                rs_name = agent_to_rs[row[AGENT_REF_KEY_NAME]["name"]]
                res = await server_client.post(server_name=rs_name, url_path="/verify", json=row)
                try:
                    await raise_for_status(res)
                except Exception:
                    if is_global_aiohttp_client_request_debug_enabled():
                        print(
                            "[rollout_reverification] /verify failed "
                            f"status={getattr(res, 'status', None)} "
                            f"row={json.dumps(_rollout_verify_debug_summary(row, rs_name), sort_keys=True)}",
                            flush=True,
                        )
                    raise
                return row, await get_response_json(res)

        return tqdm.as_completed(
            map(_post_subroutine, payloads),
            desc="Collecting reverification results",
            miniters=10,
            total=len(payloads),
            maxinterval=60,
        )

    def map_input_and_rollout_rows_to_payload(self, input_row: Dict, rollout_row: Dict) -> Dict:
        return input_row | {"response": rollout_row["response"]}

    def _prepare_payloads_from_config(self, config: RolloutReverificationConfig) -> List[Dict]:
        # to do - what about agent name?

        materialized_inputs_jsonl_fpath = resolve_input_path(config.materialized_inputs_jsonl_fpath)
        rollouts_jsonl_fpath = resolve_input_path(config.rollouts_jsonl_fpath)
        # to do - add validation for output_fpath and failures_fpath

        payloads = [
            self.map_input_and_rollout_rows_to_payload(ir, rr)
            for ir, rr in self._yield_inputs_and_rollouts_paired(materialized_inputs_jsonl_fpath, rollouts_jsonl_fpath)
        ]  # can be written better
        return payloads

    async def run_from_config(self, config: RolloutReverificationConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)

        # to do - move to other function 

        payloads_to_reverify = self._prepare_payloads_from_config(config)
        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Verifying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        from nemo_gym.rollout_collection import _failures_path_for

        output_fpath.parent.mkdir(exist_ok=True, parents=True)
        failures_fpath = _failures_path_for(output_fpath)

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in payloads_to_reverify)
        results_file = output_fpath.open("ab")
        failures_file = failures_fpath.open("ab")
        rows: List[Dict] = []
        results: List[Dict] = []
        result_strs: List[List[str]] = []
        # to do - get server name from config
        for future in self._run_reverification_payloads(payloads_to_reverify,semaphore=semaphore):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
            if SKILLS_REF_KEY_NAME in row:
                result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]

            no_persist = bool(result.get(NG_NO_PERSIST_KEY))
            failure_class = result.get(NG_FAILURE_CLASS_KEY)

            rows.append(row)
            results.append(result)
            serialized = orjson.dumps(result)
            result_strs.append(serialized)

            if no_persist:
                # kill_shaped: don't write anywhere. Set-difference on resume
                # naturally re-dispatches; per-task timeout bounds wallclock.
                pass
            elif failure_class is not None:
                # Non-kill_shaped failure → sidecar. The aggregator only reads
                # the main jsonl, so this keeps win-rate uncontaminated.
                failures_file.write(serialized + b"\n")
                failures_file.flush()
            else:
                # Success → main jsonl.
                results_file.write(serialized + b"\n")
                results_file.flush()

            counts_left[row[AGENT_REF_KEY_NAME]["name"]] -= 1
            if counts_left[row[AGENT_REF_KEY_NAME]["name"]] <= 0:
                counts_left.pop(row[AGENT_REF_KEY_NAME]["name"])

            current_pct = 100 * len(results) / len(payloads_to_reverify)
            if pcts_to_print and current_pct >= pcts_to_print[0]:
                while pcts_to_print and current_pct >= pcts_to_print[0]:
                    pcts_to_print.pop(0)

                top_left = counts_left.most_common(5)  # Fix to top 3 for now.
                if top_left:
                    top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                    # Use tqdm.write here so we can print properly with tqdm being used.
                    tqdm.write(f"Examples left:\n{top_left_str}")

        results_file.close()
        failures_file.close()
        # to do - add upload to wandb later
        if config.upload_rollouts_to_wandb and get_wandb_run():  # pragma: no cover
            print("Uploading rollouts to W&B. This may take a few minutes if your data is large.")
            get_wandb_run().log({"Rollouts": Table(data=result_strs, columns=["Rollout"])})
        del result_strs

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            aggregate_metrics_fpath = await self._call_aggregate_metrics(results, rows, output_fpath)

        print(f"""Finished rollout collection! View results at:
        Re-verified rollouts: {output_fpath}
        Aggregate metrics: {aggregate_metrics_fpath}""")

        return results

    async def _call_aggregate_metrics(
        self,
        results: List[Dict],
        rows: List[Dict],
        output_fpath: Path,
    ) -> Optional[Path]:
        """Call /aggregate_metrics on each resource server after rollouts complete.

        Writes a single _aggregate_metrics.json with one entry per agent (same shape
        as the old _agent_metrics.json). Returns the file path.
        """
        if not results:
            return None

        server_client = self.setup_resource_server_client()
        agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)
        # Group results by agent name
        agent_results: Dict[str, List[Dict]] = {}
        for row, result in zip(rows, results):
            agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            if not agent_name:
                continue
            agent_results.setdefault(agent_name, []).append(result)

        async def _fetch_agent_metrics(agent_name: str, agent_result_list: List[Dict]) -> Dict:
            # Strip heavyweight fields before sending, but preserve response.usage
            stripped = []
            for r in agent_result_list:
                entry = {k: v for k, v in r.items() if k not in ("response", "responses_create_params")}
                usage = (r.get("response") or {}).get("usage")
                if usage:
                    entry["response"] = {"usage": usage}
                stripped.append(entry)

            agg_request = AggregateMetricsRequest(verify_responses=stripped)
            agg_response = await server_client.post(
                server_name=agent_to_rs[agent_name],
                url_path="/aggregate_metrics",
                json=agg_request,
            )

            await raise_for_status(agg_response)
            agg_result = AggregateMetrics.model_validate(await get_response_json(agg_response))

            agent_entry = {
                AGENT_REF_KEY_NAME: {"name": agent_name},
                "agent_metrics": agg_result.agent_metrics,
                "key_metrics": agg_result.key_metrics,
                "group_level_metrics": agg_result.group_level_metrics,
            }
            return agent_entry

        all_agent_metrics: List[Dict] = []
        tasks = [_fetch_agent_metrics(name, results_list) for name, results_list in agent_results.items()]
        for coro in asyncio.as_completed(tasks):
            agent_entry = await coro
            all_agent_metrics.append(agent_entry)

            agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
            key_metrics = agent_entry.get("key_metrics", {})
            print(f"\nKey metrics for {agent_name}:\n" + json.dumps(key_metrics, indent=4))

        primitive_types = (bool, int, float, str, type(None))
        metrics_to_log = dict()
        for agent_entry in all_agent_metrics:
            agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
            metrics_to_log.update(
                {
                    f"{agent_name}/{k}": v
                    for k, v in agent_entry["agent_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )
            metrics_to_log.update(
                {
                    f"key_metrics/{agent_name}/{k}": v
                    for k, v in agent_entry["key_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )
        # to do - check that wandb works
        if get_wandb_run():  # pragma: no cover
            get_wandb_run().log(metrics_to_log)

        # Write single file with all agents
        metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
        metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

        return metrics_fpath


