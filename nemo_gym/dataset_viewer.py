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
import json
from typing import Any, Dict, List

from gradio import JSON, Blocks, Chatbot, ChatMessage, Dropdown
from gradio.components.chatbot import MetadataDict
from openai.types.responses.response_input_param import (
    EasyInputMessageParam,
    FunctionCallOutput,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseReasoningItemParam,
)
from pydantic import ConfigDict, Field
from tqdm.auto import tqdm

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.server_utils import get_global_config_dict
from nemo_gym.train_data_utils import (
    DatasetMetrics,
    aggregate_other_metrics,
    compute_sample_metrics,
    postprocess_other_metrics,
)


class DatasetViewerVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def format_function_call_output(m: FunctionCallOutput) -> List[ChatMessage]:
    content = m["output"]
    try:
        content = f"""```json
{json.dumps(json.loads(content), indent=4)}
```"""
    except json.JSONDecodeError:  # pragma: no cover
        pass
    return [
        ChatMessage(
            content=content,
            role="assistant",
            metadata=MetadataDict(title=f"Function call output (tool call ID `{m['call_id']}`)", status="done"),
        )
    ]


def format_function_call(m: ResponseFunctionToolCallParam) -> List[ChatMessage]:
    name = m["name"]
    arguments = json.loads(m["arguments"])
    content = f"""### Function call name `{name}`
```json
{json.dumps(arguments, indent=4)}
```"""
    return [
        ChatMessage(
            content=content,
            role="assistant",
            metadata=MetadataDict(title=f"Function call - `{name}` (tool call ID `{m['call_id']}`)", status="done"),
        )
    ]


def format_reasoning(m: ResponseReasoningItemParam) -> List[ChatMessage]:
    return [
        ChatMessage(
            content=summary_obj["text"],
            role="assistant",
            metadata=MetadataDict(title="Reasoning", status="done"),
        )
        for summary_obj in m["summary"]
    ]


def format_message(m: EasyInputMessageParam) -> List[ChatMessage]:
    content = m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}]
    match m["role"]:
        case "user":
            return [
                ChatMessage(
                    content=content_obj["text"],
                    role="user",
                    metadata=MetadataDict(title="User message", status="done"),
                )
                for content_obj in content
            ]
        case "system" | "developer":
            return [
                ChatMessage(
                    content=content_obj["text"],
                    role="assistant",
                    metadata=MetadataDict(title="System message", status="done"),
                )
                for content_obj in content
            ]
        case "assistant":
            return [
                ChatMessage(
                    content=content_obj["text"],
                    role="assistant",
                    metadata=MetadataDict(title="Assistant message", status="done"),
                )
                for content_obj in content
            ]
        case _:  # pragma: no cover
            raise NotImplementedError(f"Unrecognized role for message: `{m['role']}`")


def convert_single_message(m: ResponseInputItemParam) -> List[ChatMessage]:
    if not m.get("type") and m.get("role"):
        m["type"] = "message"

    match m["type"]:
        case "function_call_output":  # "tool"
            return format_function_call_output(m)
        case "function_call":  # "assistant tool call"
            return format_function_call(m)
        case "message":  # "assistant chat"
            return format_message(m)
        case "reasoning":  # "assistant reasoning"
            return format_reasoning(m)
        case _:  # pragma: no cover
            raise NotImplementedError(f"Unsupported message type: {m}")


def rollout_to_messages(create_params: dict, response: dict) -> List[ChatMessage]:
    messages = []
    sampling_params = create_params.copy()
    sampling_params.pop("input")
    sampling_params.pop("tools", None)
    messages.append(
        ChatMessage(
            content=f"""```json
{json.dumps(sampling_params, indent=4)}
```""",
            role="assistant",
            metadata=MetadataDict(title="Sampling params", status="done"),
        )
    )

    if create_params.get("tools"):
        messages.append(
            ChatMessage(
                content=f"""```json
{json.dumps(create_params.get("tools"), indent=4)}
```""",
                role="assistant",
                metadata=MetadataDict(title="Tools", status="done"),
            )
        )

    input = create_params["input"]
    if isinstance(input, str):
        input = [{"role": "user", "content": input}]
    turn = 0
    step = 0
    for m in input + response["output"]:
        if m.get("role") == "user":
            turn += 1
            step = 0
        if m.get("type") == "function_call":
            step += 1

        for message in convert_single_message(m):
            message.metadata["title"] = f"Turn {turn} Step {step} - {message.metadata['title']}"
            messages.append(message)

    return messages


def extra_info_to_messages(d: DatasetViewerVerifyResponse) -> List[ChatMessage]:
    messages = []
    for k, v in d.items():
        if not isinstance(v, (int, float, bool, str, list, dict)):  # pragma: no cover
            continue

        str_v = (
            v
            if isinstance(v, str)
            else f"""```json
{json.dumps(v, indent=4)}
```"""
        )
        message = ChatMessage(
            content=str_v,
            role="user",
            metadata=MetadataDict(title=f"Metadata - {k}", status="done"),
        )
        messages.append(message)

    return messages


class JsonlDatasetViewerConfig(BaseNeMoGymCLIConfig):
    jsonl_fpath: str = Field(description="Filepath to a local jsonl file to view.")


def get_aggregate_metrics(data: List[DatasetViewerVerifyResponse]) -> Dict[str, Any]:
    dataset_metrics = DatasetMetrics()
    other_metrics = {}

    for line in data:
        line = json.dumps(line.model_dump())
        metrics, is_offending = compute_sample_metrics(line)
        if not is_offending:
            dataset_metrics.add(metrics)

        sample_dict = json.loads(line)
        aggregate_other_metrics(other_metrics, sample_dict)

    postprocess_other_metrics(dataset_metrics, other_metrics)

    aggregate_metrics = dataset_metrics.aggregate()
    aggregate_metrics_dict = aggregate_metrics.model_dump(by_alias=True)
    return aggregate_metrics_dict


def build_jsonl_dataset_viewer(config: JsonlDatasetViewerConfig) -> Blocks:
    with open(config.jsonl_fpath) as f:
        data = list(
            tqdm(
                map(DatasetViewerVerifyResponse.model_validate_json, f),
                desc="Loading data",
            )
        )

    choices = [(f"Sample {i + 1} - Responses ID {d.response.id}", i) for i, d in enumerate(data)]

    def select_item(value: int):
        d = data[value]
        return extra_info_to_messages(d.model_dump()) + rollout_to_messages(
            d.responses_create_params.model_dump(), d.response.model_dump()
        )

    CSS = """
    footer {
        visibility: hidden;
    }
    """
    with Blocks(analytics_enabled=False, css=CSS) as demo:
        aggregate_dicts = get_aggregate_metrics(data)
        JSON(value=aggregate_dicts, label="Aggregate Metrics", open=False)

        item_dropdown = Dropdown(choices=choices, value=0, label="Samples")
        chatbot = Chatbot(
            value=select_item(0),
            type="messages",
            height="80vh",
            layout="panel",
            label="Rollout",
        )
        item_dropdown.select(fn=select_item, inputs=item_dropdown, outputs=chatbot, show_api=False)

    return demo


def main():  # pragma: no cover
    config = JsonlDatasetViewerConfig.model_validate(get_global_config_dict())
    demo = build_jsonl_dataset_viewer(config)
    demo.launch(enable_monitoring=False)
