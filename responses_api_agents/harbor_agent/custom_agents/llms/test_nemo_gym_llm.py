from unittest.mock import Mock
from pathlib import Path

import pytest
import requests

from harbor.llms.base import ContextLengthExceededError
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


@pytest.mark.asyncio
async def test_nemo_gym_llm_extracts_openai_shape(monkeypatch):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base="http://localhost:8000/v1",
        collect_rollout_details=True,
    )

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "hello"},
                "provider_specific_fields": {"token_ids": [7, 8]},
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                "finish_reason": "stop",
            }
        ],
        "prompt_token_ids": [1, 2, 3],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_response)

    response = await llm.call(prompt="hello")
    assert response.content == "hello"
    assert response.prompt_token_ids == [1, 2, 3]
    assert response.completion_token_ids == [7, 8]
    assert response.logprobs == [-0.1, -0.2]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.cache_tokens == 4


@pytest.mark.asyncio
async def test_nemo_gym_llm_extracts_nemo_proxy_shape(monkeypatch):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base="http://localhost:8000/v1",
        collect_rollout_details=True,
    )

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "proxy output",
                    "prompt_token_ids": [11, 12],
                    "generation_token_ids": ["token_id:13", "token_id:14"],
                    "generation_log_probs": [-0.3, -0.4],
                },
                "finish_reason": "stop",
            }
        ],
    }
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_response)

    response = await llm.call(prompt="hello")
    assert response.content == "proxy output"
    assert response.prompt_token_ids == [11, 12]
    assert response.completion_token_ids == [13, 14]
    assert response.logprobs == [-0.3, -0.4]


@pytest.mark.asyncio
async def test_nemo_gym_llm_context_error_translation(monkeypatch):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base="http://localhost:8000/v1",
    )

    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "maximum context length exceeded"
    http_error = requests.HTTPError("400 bad request")
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_response)
    with pytest.raises(ContextLengthExceededError):
        await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_nemo_gym_llm_no_rollout_details_for_openai_model(monkeypatch):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base="http://localhost:8000/v1",
        collect_rollout_details=True,
    )

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "plain output"},
                "finish_reason": "stop",
            }
        ],
    }
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_response)

    response = await llm.call(prompt="hello")
    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None
    assert response.logprobs is None


@pytest.mark.asyncio
async def test_nemo_gym_llm_serializes_path_kwargs(monkeypatch):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base="http://localhost:8000/v1",
    )

    captured_payload: dict = {}

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }

    def _mock_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs["json"]
        return mock_response

    monkeypatch.setattr("requests.post", _mock_post)

    response = await llm.call(
        prompt="hello",
        metadata={"workspace_path": Path("/tmp/workspace")},
        files=[Path("/tmp/a.txt"), Path("/tmp/b.txt")],
    )

    assert response.content == "ok"
    assert captured_payload["metadata"]["workspace_path"] == "/tmp/workspace"
    assert captured_payload["files"] == ["/tmp/a.txt", "/tmp/b.txt"]


@pytest.mark.parametrize(
    ("api_base", "expected_endpoint"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_nemo_gym_llm_chat_completions_endpoint(api_base, expected_endpoint):
    llm = NemoGymLLM(
        model_name="test-model",
        api_base=api_base,
    )
    assert llm._chat_completions_endpoint() == expected_endpoint
