# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from responses_api_agents.parallel_reasoning_agent.app import (
    ModelServerRef,
    ParallelReasoning,
    ParallelReasoningConfig,
    ResourcesServerRef,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ParallelReasoningConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            num_executor=16,
            num_reducer=1,
            max_output_tokens=None,
            use_reducer=True,
            return_reducer_only=False,
            reducer_type="genselect_tournament",
            reduce_across="all",
            tournament_group_size=4,
        )
        ParallelReasoning(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = ParallelReasoningConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            num_executor=1,
            num_reducer=1,
            max_output_tokens=None,
            use_reducer=False,
            return_reducer_only=False,
            reducer_type="genselect_tournament",
            reduce_across="all",
            tournament_group_size=4,
        )
        server = ParallelReasoning(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_a90cfc29f75b4a5cb5366c1a3512a4a8",
            "created_at": 1760662672.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {"stage": "executor"},
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "object": "response",
            "output": [
                {
                    "id": "msg_7e5a0f3644ab46aba281697ee5378f3c",
                    "content": [
                        {
                            "annotations": [],
                            "text": "We are given the expression:\n\n$$\n(75 + 117i)z + \\frac{96 + 144i}{z}\n$$\n\nand we are told that $ z $ is a complex number with $ |z| = 4 $. We are to **maximize the real part** of this expression.\n\n---\n\n### Step 1: Use the condition $ |z| = 4 $\n\nSince $ |z| = 4 $, we know that $ z \\overline{z} = |z|^2 = 16 $.  \nThis implies $ \\frac{1}{z} = \\frac{\\overline{z}}{|z|^2} = \\frac{\\overline{z}}{16} $.  \nSo, we can write:\n\n$$\n\\frac{1}{z} = \\frac{\\overline{z}}{16}\n$$\n\nThis will help us rewrite the expression in terms of $ z $ and $ \\overline{z} $.\n\n---\n\n### Step 2: Let $ z = 4e^{i\\theta} $\n\nSince $ |z| = 4 $, we can write $ z = 4e^{i\\theta} $, where $ \\theta \\in [0, 2\\pi) $.  \nThen $ \\frac{1}{z} = \\frac{1}{4}e^{-i\\theta} $.\n\nNow plug into the expression:\n\n$$\n(75 + 117i)z + \\frac{96 + 144i}{z} = (75 + 117i)(4e^{i\\theta}) + (96 + 144i)\\left(\\frac{1}{4}e^{-i\\theta}\\right)\n$$\n\nSimplify:\n\n$$\n= 4(75 + 117i)e^{i\\theta} + \\frac{1}{4}(96 + 144i)e^{-i\\theta}\n$$\n\nFactor out constants:\n\n$$\n= 4(75 + 117i)e^{i\\theta} + \\frac{1}{4}(96 + 144i)e^{-i\\theta}\n$$\n\nLet us denote:\n\n- $ A = 75 + 117i $\n- $ B = 96 + 144i $\n\nSo the expression becomes:\n\n$$\n4A e^{i\\theta} + \\frac{1}{4}B e^{-i\\theta}\n$$\n\nWe are to find the **maximum of the real part** of this expression over $ \\theta \\in [0, 2\\pi) $.\n\n---\n\n### Step 3: Take real part\n\nLet $ w(\\theta) = 4A e^{i\\theta} + \\frac{1}{4}B e^{-i\\theta} $\n\nWe compute $ \\text{Re}(w(\\theta)) $:\n\n$$\n\\text{Re}(w(\\theta)) = 4 \\text{Re}(A e^{i\\theta}) + \\frac{1}{4} \\text{Re}(B e^{-i\\theta})\n$$\n\nNow, recall that for a complex number $ c = a + bi $, $ \\text{Re}(c e^{i\\theta}) = a \\cos\\theta - b \\sin\\theta $, and $ \\text{Re}(c e^{-i\\theta}) = a \\cos\\theta + b \\sin\\theta $.\n\nLet’s compute $ A = 75 + 117i $, so:\n\n- $ \\text{Re}(A e^{i\\theta}) = 75\\cos\\theta - 117\\sin\\theta $\n\nSimilarly, $ B = 96 + 144i $, so:\n\n- $ \\text{Re}(B e^{-i\\theta}) = 96\\cos\\theta + 144\\sin\\theta $\n\nTherefore:\n\n$$\n\\text{Re}(w(\\theta)) = 4(75\\cos\\theta - 117\\sin\\theta) + \\frac{1}{4}(96\\cos\\theta + 144\\sin\\theta)\n$$\n\nNow expand:\n\n$$\n= 4 \\cdot 75 \\cos\\theta - 4 \\cdot 117 \\sin\\theta + \\frac{1}{4} \\cdot 96 \\cos\\theta + \\frac{1}{4} \\cdot 144 \\sin\\theta\n$$\n\nCompute each term:\n\n- $ 4 \\cdot 75 = 300 $\n- $ 4 \\cdot 117 = 468 $\n- $ \\frac{1}{4} \\cdot 96 = 24 $\n- $ \\frac{1}{4} \\cdot 144 = 36 $\n\nSo:\n\n$$\n\\text{Re}(w(\\theta)) = (300 + 24)\\cos\\theta + (-468 + 36)\\sin\\theta = 324\\cos\\theta - 432\\sin\\theta\n$$\n\n---\n\n### Step 4: Maximize $ 324\\cos\\theta - 432\\sin\\theta $\n\nThis is a standard trigonometric expression of the form $ R\\cos(\\theta + \\phi) $, and its maximum value is $ \\sqrt{a^2 + b^2} $ when written as $ a\\cos\\theta + b\\sin\\theta $, but here we have:\n\n$$\n324\\cos\\theta - 432\\sin\\theta\n$$\n\nThis can be written as:\n\n$$\nR \\cos(\\theta + \\phi)\n$$\n\nwhere $ R = \\sqrt{324^2 + 432^2} $\n\nCompute $ R $:\n\nFirst, factor out common terms:\n\nNote: 324 and 432 are both divisible by 108.\n\n- $ 324 = 108 \\cdot 3 $\n- $ 432 = 108 \\cdot 4 $\n\nSo:\n\n$$\nR = \\sqrt{(108 \\cdot 3)^2 + (108 \\cdot 4)^2} = 108 \\sqrt{3^2 + 4^2} = 108 \\sqrt{9 + 16} = 108 \\sqrt{25} = 108 \\cdot 5 = 540\n$$\n\nTherefore, the maximum value of $ 324\\cos\\theta - 432\\sin\\theta $ is $ \\boxed{540} $\n\n---\n\n### Final Answer:\n\nAnswer: 540",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }

        mock_response_object = MagicMock()
        mock_response_object.json = AsyncMock(return_value=mock_response_data)
        server.server_client.post = AsyncMock(return_value=mock_response_object)

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = [
            {
                "id": "resp_a90cfc29f75b4a5cb5366c1a3512a4a8",
                "created_at": 1760662672.0,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {"stage": "executor"},
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "object": "response",
                "output": [
                    {
                        "id": "msg_7e5a0f3644ab46aba281697ee5378f3c",
                        "content": [
                            {
                                "annotations": [],
                                "text": "We are given the expression:\n\n$$\n(75 + 117i)z + \\frac{96 + 144i}{z}\n$$\n\nand we are told that $ z $ is a complex number with $ |z| = 4 $. We are to **maximize the real part** of this expression.\n\n---\n\n### Step 1: Use the condition $ |z| = 4 $\n\nSince $ |z| = 4 $, we know that $ z \\overline{z} = |z|^2 = 16 $.  \nThis implies $ \\frac{1}{z} = \\frac{\\overline{z}}{|z|^2} = \\frac{\\overline{z}}{16} $.  \nSo, we can write:\n\n$$\n\\frac{1}{z} = \\frac{\\overline{z}}{16}\n$$\n\nThis will help us rewrite the expression in terms of $ z $ and $ \\overline{z} $.\n\n---\n\n### Step 2: Let $ z = 4e^{i\\theta} $\n\nSince $ |z| = 4 $, we can write $ z = 4e^{i\\theta} $, where $ \\theta \\in [0, 2\\pi) $.  \nThen $ \\frac{1}{z} = \\frac{1}{4}e^{-i\\theta} $.\n\nNow plug into the expression:\n\n$$\n(75 + 117i)z + \\frac{96 + 144i}{z} = (75 + 117i)(4e^{i\\theta}) + (96 + 144i)\\left(\\frac{1}{4}e^{-i\\theta}\\right)\n$$\n\nSimplify:\n\n$$\n= 4(75 + 117i)e^{i\\theta} + \\frac{1}{4}(96 + 144i)e^{-i\\theta}\n$$\n\nFactor out constants:\n\n$$\n= 4(75 + 117i)e^{i\\theta} + \\frac{1}{4}(96 + 144i)e^{-i\\theta}\n$$\n\nLet us denote:\n\n- $ A = 75 + 117i $\n- $ B = 96 + 144i $\n\nSo the expression becomes:\n\n$$\n4A e^{i\\theta} + \\frac{1}{4}B e^{-i\\theta}\n$$\n\nWe are to find the **maximum of the real part** of this expression over $ \\theta \\in [0, 2\\pi) $.\n\n---\n\n### Step 3: Take real part\n\nLet $ w(\\theta) = 4A e^{i\\theta} + \\frac{1}{4}B e^{-i\\theta} $\n\nWe compute $ \\text{Re}(w(\\theta)) $:\n\n$$\n\\text{Re}(w(\\theta)) = 4 \\text{Re}(A e^{i\\theta}) + \\frac{1}{4} \\text{Re}(B e^{-i\\theta})\n$$\n\nNow, recall that for a complex number $ c = a + bi $, $ \\text{Re}(c e^{i\\theta}) = a \\cos\\theta - b \\sin\\theta $, and $ \\text{Re}(c e^{-i\\theta}) = a \\cos\\theta + b \\sin\\theta $.\n\nLet’s compute $ A = 75 + 117i $, so:\n\n- $ \\text{Re}(A e^{i\\theta}) = 75\\cos\\theta - 117\\sin\\theta $\n\nSimilarly, $ B = 96 + 144i $, so:\n\n- $ \\text{Re}(B e^{-i\\theta}) = 96\\cos\\theta + 144\\sin\\theta $\n\nTherefore:\n\n$$\n\\text{Re}(w(\\theta)) = 4(75\\cos\\theta - 117\\sin\\theta) + \\frac{1}{4}(96\\cos\\theta + 144\\sin\\theta)\n$$\n\nNow expand:\n\n$$\n= 4 \\cdot 75 \\cos\\theta - 4 \\cdot 117 \\sin\\theta + \\frac{1}{4} \\cdot 96 \\cos\\theta + \\frac{1}{4} \\cdot 144 \\sin\\theta\n$$\n\nCompute each term:\n\n- $ 4 \\cdot 75 = 300 $\n- $ 4 \\cdot 117 = 468 $\n- $ \\frac{1}{4} \\cdot 96 = 24 $\n- $ \\frac{1}{4} \\cdot 144 = 36 $\n\nSo:\n\n$$\n\\text{Re}(w(\\theta)) = (300 + 24)\\cos\\theta + (-468 + 36)\\sin\\theta = 324\\cos\\theta - 432\\sin\\theta\n$$\n\n---\n\n### Step 4: Maximize $ 324\\cos\\theta - 432\\sin\\theta $\n\nThis is a standard trigonometric expression of the form $ R\\cos(\\theta + \\phi) $, and its maximum value is $ \\sqrt{a^2 + b^2} $ when written as $ a\\cos\\theta + b\\sin\\theta $, but here we have:\n\n$$\n324\\cos\\theta - 432\\sin\\theta\n$$\n\nThis can be written as:\n\n$$\nR \\cos(\\theta + \\phi)\n$$\n\nwhere $ R = \\sqrt{324^2 + 432^2} $\n\nCompute $ R $:\n\nFirst, factor out common terms:\n\nNote: 324 and 432 are both divisible by 108.\n\n- $ 324 = 108 \\cdot 3 $\n- $ 432 = 108 \\cdot 4 $\n\nSo:\n\n$$\nR = \\sqrt{(108 \\cdot 3)^2 + (108 \\cdot 4)^2} = 108 \\sqrt{3^2 + 4^2} = 108 \\sqrt{9 + 16} = 108 \\sqrt{25} = 108 \\cdot 5 = 540\n$$\n\nTherefore, the maximum value of $ 324\\cos\\theta - 432\\sin\\theta $ is $ \\boxed{540} $\n\n---\n\n### Final Answer:\n\nAnswer: 540",
                                "type": "output_text",
                                "logprobs": None,
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": True,
                "temperature": None,
                "tool_choice": "auto",
                "tools": [],
                "top_p": None,
                "background": None,
                "max_output_tokens": None,
                "max_tool_calls": None,
                "previous_response_id": None,
                "prompt": None,
                "reasoning": None,
                "service_tier": None,
                "status": None,
                "text": None,
                "top_logprobs": None,
                "truncation": None,
                "usage": None,
                "user": None,
            }
        ]
        assert expected_responses_dict == actual_responses_dict
