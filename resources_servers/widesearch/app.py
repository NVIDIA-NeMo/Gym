# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dateparser
import pandas as pd
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


class WideSearchConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class WideSearchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str
    query: str
    evaluation: dict[str, Any]
    gold_answer: list[dict[str, Any]]
    language: str


class WideSearchVerifyRequest(WideSearchRunRequest, BaseVerifyRequest):
    pass


class WideSearchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    score: float
    precision_by_row: float
    recall_by_row: float
    f1_by_row: float
    precision_by_item: float
    recall_by_item: float
    f1_by_item: float
    evaluation_message: str


def response_text(response: NeMoGymResponse) -> str:
    texts = []
    for output in response.output:
        if output.type == "message" and output.role == "assistant":
            texts.extend(part.text for part in output.content if getattr(part, "text", None))
    return "\n".join(texts).strip()


def norm_column(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def parse_markdown_json(text: str) -> dict[str, Any]:
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not matches:
        return {}
    try:
        result = json.loads(matches[-1])
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_dataframe(text: str) -> pd.DataFrame | None:
    matches = re.findall(r"```markdown(.*?)```", text, re.DOTALL)
    if not matches:
        table = re.findall(r"((?:^\s*\|.*\|\s*$\n?)+)", text, re.MULTILINE)
        matches = table[-1:] if table else []
    if not matches:
        return None
    lines = []
    for line in matches[0].strip().splitlines():
        if "|" in line and not set(line.strip()).issubset(set("|- :")):
            lines.append("|".join(part.strip() for part in line.split("|")))
    if not lines:
        return None
    frame = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def preprocess(value: Any, name: str) -> str:
    value = str(value)
    if name == "norm_str":
        return value.lower().strip().replace(" ", "").replace("*", "")
    if name == "extract_number":
        numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", value.replace(",", ""))
        return numbers[0] if numbers else "NULL"
    if name == "norm_date":
        parsed = dateparser.parse(value, settings={"PREFER_DAY_OF_MONTH": "first"})
        return parsed.strftime("%Y-%m-%d") if parsed else value
    raise ValueError(f"unsupported preprocess function: {name}")


def deterministic_metric(response: str, target: str, name: str, criterion: Any) -> float:
    if name == "exact_match":
        return float(response.lower() == target.lower())
    if name == "url_match":
        pattern = re.compile(r"https?://[^\s]+")
        response_hosts = {urlparse(url).netloc for url in pattern.findall(response)}
        target_hosts = {urlparse(url).netloc for url in pattern.findall(target)}
        return float(response_hosts == target_hosts)
    if name == "number_near":
        try:
            response_number = float(response.rstrip("%")) / (100 if "%" in response else 1)
            target_number = float(target.rstrip("%")) / (100 if "%" in target else 1)
        except ValueError:
            return float(response == target)
        return float(abs(response_number - target_number) <= abs(target_number) * float(criterion))
    if name == "date_near":
        try:
            response_date, target_date = dateparser.parse(response), dateparser.parse(target)
        except (TypeError, ValueError, OverflowError):
            response_date, target_date = None, None
        if response_date is None or target_date is None:
            return float(response_date is None and target_date is None)
        return float(abs((response_date - target_date).days) <= 31)
    raise ValueError(f"unsupported metric: {name}")


class WideSearchServer(SimpleResourcesServer):
    config: WideSearchConfig

    def model_post_init(self, context: Any) -> None:
        prompts = Path(__file__).with_name("prompts")
        self._alignment_prompt = (prompts / "alignment.txt").read_text()
        self._column_judge_prompt = (prompts / "column_judge.txt").read_text()
        super().model_post_init(context)

    async def _judge(self, prompt: str) -> str:
        params = self.config.judge_responses_create_params.model_copy(
            update={"input": [NeMoGymEasyInputMessage(role="user", content=prompt)]}, deep=True
        )
        response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        return response_text(response)

    async def _align(self, values: list[str], references: list[str]) -> dict[str, str]:
        if not values or not references:
            return {}
        if set(values).issubset(references):
            return {value: value for value in values}
        prompt = self._alignment_prompt.format(response=values, reference=references)
        return {str(key): str(value) for key, value in parse_markdown_json(await self._judge(prompt)).items()}

    async def _llm_scores(self, responses: list[str], targets: list[str], criterion: str) -> list[float]:
        pairs = {
            f"idx_{index}": {"response": response, "target": target}
            for index, (response, target) in enumerate(zip(responses, targets))
        }
        prompt = self._column_judge_prompt.format(criterion=criterion, response=pairs)
        result = parse_markdown_json(await self._judge(prompt))
        return [float(result.get(f"idx_{index}", 0) == 1) for index in range(len(responses))]

    async def verify(self, body: WideSearchVerifyRequest) -> WideSearchVerifyResponse:
        zeros = {
            "score": 0.0,
            "precision_by_row": 0.0,
            "recall_by_row": 0.0,
            "f1_by_row": 0.0,
            "precision_by_item": 0.0,
            "recall_by_item": 0.0,
            "f1_by_item": 0.0,
        }
        response_frame = extract_dataframe(response_text(body.response))
        if response_frame is None:
            return WideSearchVerifyResponse(
                **body.model_dump(), reward=0.0, **zeros, evaluation_message="response contains no Markdown table"
            )

        required = body.evaluation["required"]
        unique = body.evaluation["unique_columns"]
        gold = pd.DataFrame(body.gold_answer)
        gold.columns = [norm_column(column) for column in gold.columns]
        response_frame.columns = [norm_column(column) for column in response_frame.columns]
        if set(response_frame.columns) != set(required):
            response_frame.rename(
                columns=await self._align(response_frame.columns.tolist(), required),
                inplace=True,
            )
        if set(response_frame.columns) != set(required):
            return WideSearchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                **zeros,
                evaluation_message=f"required columns {required} do not match {response_frame.columns.tolist()}",
            )

        gold = gold[required].astype(str)
        response_frame = response_frame[required].astype(str)
        for column in unique:
            pipeline = body.evaluation["eval_pipeline"].get(column, {})
            if {"exact_match", "llm_judge"}.intersection(pipeline.get("metric", [])):
                mapping = await self._align(response_frame[column].tolist(), gold[column].tolist())
                response_frame[column] = response_frame[column].map(lambda value: mapping.get(value, value))
        for column, pipeline in body.evaluation["eval_pipeline"].items():
            for name in pipeline.get("preprocess", []):
                gold[column] = gold[column].map(lambda value, name=name: preprocess(value, name))
                response_frame[column] = response_frame[column].map(lambda value, name=name: preprocess(value, name))
        gold = gold.drop_duplicates(subset=unique)
        response_frame = response_frame.drop_duplicates(subset=unique)

        gold_rows = {tuple(row[column] for column in unique): row for _, row in gold.iterrows()}
        response_rows = {tuple(row[column] for column in unique): row for _, row in response_frame.iterrows()}
        matched_keys = list(gold_rows.keys() & response_rows.keys())
        scores = {key: {column: 1.0 for column in unique} for key in matched_keys}
        for column in required:
            if column in unique:
                continue
            pipeline = body.evaluation["eval_pipeline"][column]
            responses = [response_rows[key][column] for key in matched_keys]
            targets = [gold_rows[key][column] for key in matched_keys]
            column_scores = [1.0] * len(matched_keys)
            for metric in pipeline.get("metric", []):
                if metric == "llm_judge":
                    metric_scores = await self._llm_scores(responses, targets, pipeline.get("criterion", ""))
                else:
                    metric_scores = [
                        deterministic_metric(response, target, metric, pipeline.get("criterion"))
                        for response, target in zip(responses, targets)
                    ]
                column_scores = [min(current, new) for current, new in zip(column_scores, metric_scores)]
            for key, score in zip(matched_keys, column_scores):
                scores[key][column] = score

        true_rows = sum(min(row.values()) for row in scores.values())
        true_items = sum(sum(row.values()) for row in scores.values())
        predicted_rows, gold_row_count = len(response_frame), len(gold)
        predicted_items, gold_items = predicted_rows * len(required), gold_row_count * len(required)
        precision_row = true_rows / predicted_rows if predicted_rows else 0.0
        recall_row = true_rows / gold_row_count if gold_row_count else 0.0
        precision_item = true_items / predicted_items if predicted_items else 0.0
        recall_item = true_items / gold_items if gold_items else 0.0
        f1_row = 2 * precision_row * recall_row / (precision_row + recall_row) if precision_row + recall_row else 0.0
        f1_item = (
            2 * precision_item * recall_item / (precision_item + recall_item) if precision_item + recall_item else 0.0
        )
        strict = float(precision_row == recall_row == f1_row == precision_item == recall_item == f1_item == 1.0)
        return WideSearchVerifyResponse(
            **body.model_dump(),
            reward=strict,
            score=strict,
            precision_by_row=precision_row,
            recall_by_row=recall_row,
            f1_by_row=f1_row,
            precision_by_item=precision_item,
            recall_by_item=recall_item,
            f1_by_item=f1_item,
            evaluation_message="All items match perfectly."
            if strict
            else "Table is incomplete or contains mismatches.",
        )


if __name__ == "__main__":
    WideSearchServer.run_webserver()
