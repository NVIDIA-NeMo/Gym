from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import NeMoGymResponse

from app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest


class TestApp:
    def test_sanity(self) -> None:
        config = MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="")
        MCQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_correct(self) -> None:
        # Build a NeMoGymResponse with a valid OpenAI Responses shape and the assistant message including letter C
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is C.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = MCQAResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = MCQAVerifyRequest(
            responses_create_params={
                "input": [
                    {
                        "role": "user",
                        "content": "Q?\nA: optA\nB: optB\nC: optC\nD: optD",
                    },
                ],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            response=response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict_single_letter_boxed",
        )

        # strict requires boxed; plain C should fail
        result = await server.verify(verify_request)
        assert result.reward == 0.0

        # Now send boxed C (strict)
        response_boxed = NeMoGymResponse(
            id="resp_test2",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test2",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Final: \\boxed{ [C] }",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request_boxed = verify_request.model_copy(
            update={"response": response_boxed}
        )
        result2 = await server.verify(verify_request_boxed)
        assert result2.reward == 1.0

        # Lenient: allow matching option text within boxed content
        response_boxed_text = NeMoGymResponse(
            id="resp_test3",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Final: \\boxed{ optC }",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request_lenient = verify_request.model_copy(
            update={"response": response_boxed_text, "grading_mode": "lenient_boxed"}
        )
        result3 = await server.verify(verify_request_lenient)
        assert result3.reward == 1.0

        # Lenient answer colon: letter
        response_answer_colon = NeMoGymResponse(
            id="resp_test4",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test4",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Answer: c",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        verify_request_answer_colon = verify_request.model_copy(
            update={
                "response": response_answer_colon,
                "grading_mode": "lenient_answer_colon",
            }
        )
        result4 = await server.verify(verify_request_answer_colon)
        assert result4.reward == 1.0

        # Lenient answer colon: exact option text
        response_answer_colon_text = NeMoGymResponse(
            id="resp_test5",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test5",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Answer: optC",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        verify_request_answer_colon_text = verify_request.model_copy(
            update={
                "response": response_answer_colon_text,
                "grading_mode": "lenient_answer_colon",
            }
        )
        result5 = await server.verify(verify_request_answer_colon_text)
        assert result5.reward == 1.0
