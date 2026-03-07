from unittest.mock import MagicMock

from app import (
    Ether0ResourcesServer,
    Ether0ResourcesServerConfig,
    Ether0VerifyRequest,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_server() -> Ether0ResourcesServer:
    config = Ether0ResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return Ether0ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(text: str, solution: str, problem_type: str) -> Ether0VerifyRequest:
    return Ether0VerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q"}]},
        response=_make_response(text),
        verifier_metadata={
            "solution": solution,
            "problem_type": problem_type,
        },
    )


class TestVerify:
    def test_sanity(self) -> None:
        _make_server()

    async def test_str_eval_correct(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>FCC(=O)O</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_str_eval_wrong(self) -> None:
        server = _make_server()
        req = _make_request(
            "<answer>CCCCCC</answer>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_ether0_special_tokens(self) -> None:
        server = _make_server()
        req = _make_request(
            "<|think_start|>reasoning here<|think_end|><|answer_start|>FCC(=O)O<|answer_end|>",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_no_answer_tag(self) -> None:
        server = _make_server()
        req = _make_request(
            "I have no idea",
            "str_eval!:!FCC(=O)O!:!property-regression-ld50",
            "property-regression-ld50",
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer is None

    async def test_malformed_solution(self) -> None:
        server = _make_server()
        req = _make_request("<answer>CCO</answer>", "bad_format", "")
        result = await server.verify(req)
        assert result.reward == 0.0
