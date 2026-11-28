# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-Index v1.0 / v1.1 fixed-cohort resources server built on VideoMCQAResourcesServer."""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

from pydantic import PrivateAttr

from nemo_gym.client import ServerClient
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.video_mcqa.app import (
    VideoMCQAResourcesServer,
    VideoMCQAResourcesServerConfig,
    VideoMCQAVerifyRequest,
    VideoMCQAVerifyResponse,
)
from resources_servers.video_mcqa.scoring import Failure, QuestionScore
from resources_servers.video_mcqa.task import Setting, VerifierMetadata
from resources_servers.videoindex.dataset import RELEASES, VideoIndexCohort
from resources_servers.videoindex.scoring import (
    MAX_OPTIONS,
    VideoIndexAnswerPolicy,
    VideoIndexQuestionScore,
    grade_videoindex_answer,
)


class VideoIndexVerifyResponse(VideoMCQAVerifyResponse):
    """Verify response supporting both Video-Index's official rule policy and shared policies."""

    answer_policy: VideoIndexAnswerPolicy


class VideoIndexResourcesServer(VideoMCQAResourcesServer):
    """Fixed-cohort verifier for Video-Index v1.0 (840 Q) and v1.1 (144 Q) across 4 capability groups."""

    verify_response_cls: ClassVar[type[VideoMCQAVerifyResponse]] = VideoIndexVerifyResponse
    score_cls: ClassVar[type[QuestionScore]] = VideoIndexQuestionScore
    _cohort: VideoIndexCohort = PrivateAttr()

    def _load_cohort(self, manifest: Path) -> VideoIndexCohort:
        return VideoIndexCohort.model_validate_json(manifest.read_text(encoding="utf-8"))

    def _validate_cohort(self) -> None:
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return
        configured = global_config.get("num_repeats")
        if configured is None:
            return
        expected_repeats: int | None = None
        if isinstance(configured, int) and not isinstance(configured, bool):
            expected_repeats = configured
        elif isinstance(configured, Mapping):
            for key in ("videoindex_single_turn", "videoindex_agentic_opencode", "_default"):
                value = configured.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    expected_repeats = value
                    break
        if expected_repeats is not None and expected_repeats != self._cohort.num_repeats:
            raise ValueError(
                f"Configured num_repeats ({expected_repeats}) disagrees with frozen cohort num_repeats "
                f"({self._cohort.num_repeats}); re-run prepare with ++num_repeats={expected_repeats} "
                f"(or ++prepare_script_args.num_repeats={expected_repeats})."
            )

    def _validate_option_count(self, metadata: VerifierMetadata) -> None:
        if not 2 <= metadata.option_count <= MAX_OPTIONS:
            raise ValueError(f"Video-Index requires between 2 and {MAX_OPTIONS} choices")

    def _cohort_scope(self) -> str:
        return self._cohort.benchmark

    def _metric_spec(self) -> tuple[str, dict[str, dict[str, tuple[str, ...]]]]:
        metric = RELEASES[self._cohort.benchmark].metric_name
        return metric, self._cohort.task_subsets

    def _grade_rule(
        self,
        final_text: str | None,
        metadata: VerifierMetadata,
        *,
        setting: Setting,
        answer_policy: str,
        failure: Failure | None = None,
    ) -> QuestionScore:
        return grade_videoindex_answer(
            final_text, metadata, setting=setting, answer_policy=answer_policy, failure=failure
        )

    async def verify(self, body: VideoMCQAVerifyRequest) -> VideoIndexVerifyResponse:
        """Score final assistant text with Video-Index's rule or configured answer judge."""
        result = await super().verify(body)
        assert isinstance(result, VideoIndexVerifyResponse)
        return result


def _fixture_server() -> VideoIndexResourcesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return VideoIndexResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="videoindex",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cohort_manifest=Path(__file__).parent / "tests" / "verifier_cohort.json",
        ),
        server_client=client,
    )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=VideoMCQAVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VideoIndexResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VideoIndexResourcesServer.run_webserver()
