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

"""Customer service multi-turn environment with separate tools for policy and user agent.

Policy agent = support agent. Sees only conversation text.
User agent = simulated customer with private tools. Policy agent never sees the
user agent's tool calls or results.

All scenario data comes from the JSONL, including user model tool schemas.
Use scripts/generate_data.py to create datasets.

verifier_metadata fields:
  customer:             dict    name, email
  order:                dict    order_id, product_name, price, status, days_since_order
  issue_type:           str     refund, order_status, wrong_item, damaged, cancel
  opener:               str     customer's opening message
  resolution_keywords:  list    keywords that indicate correct resolution
  policies:             dict    refund and cancel policy text
  user_tools:           list    tool schemas for the user model
"""

import json
from typing import Any, Dict, Optional

from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.example_gymnasium import GymnasiumServer, extract_text


class CustomerServiceEnv(GymnasiumServer):
    user_model_server: str = "customer_service_user_model"
    session_state: Dict[str, Any] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        self.session_state[session_id] = metadata
        return metadata.get("opener"), {}

    def _execute_user_tool(self, tool_name: str, args: dict, scenario: dict) -> str:
        if tool_name == "lookup_order":
            order = scenario.get("order", {})
            if args.get("order_id") == order.get("order_id"):
                return json.dumps(order)
            return json.dumps({"error": "Order not found"})
        if tool_name == "check_account":
            return json.dumps(scenario.get("customer", {}))
        if tool_name == "get_policy":
            policies = scenario.get("policies", {})
            return policies.get(args.get("policy_type", ""), "Policy not found.")
        return f"Unknown tool: {tool_name}"

    async def _get_user_reply(self, agent_text: str, scenario: dict) -> str:
        customer = scenario.get("customer", {})
        system = (
            f"You are {customer.get('name', 'a customer')} contacting support. "
            f"Your email is {customer.get('email', '')}. "
            f"You have a {scenario.get('issue_type', '')} issue. "
            f"Use your tools to look up details when needed. "
            f"If the agent has resolved your issue, say 'Thanks, that resolves my issue.' "
            f"Keep responses short and natural."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": agent_text},
        ]

        for _ in range(3):
            try:
                resp = await self.server_client.post(
                    server_name=self.user_model_server,
                    url_path="/v1/responses",
                    json={"input": messages, "tools": scenario.get("user_tools", [])},
                )
                await raise_for_status(resp)
                data = await get_response_json(resp)
            except Exception:
                return "Could you repeat that?"

            outputs = data.get("output", [])
            tool_calls = [o for o in outputs if o.get("type") == "function_call"]

            if not tool_calls:
                for o in outputs:
                    if o.get("type") == "message":
                        for c in o.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                return c.get("text", "")
                return "I see."

            for tc in tool_calls:
                args = json.loads(tc.get("arguments", "{}"))
                result = self._execute_user_tool(tc["name"], args, scenario)
                messages.append(tc)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.get("call_id", ""),
                        "output": result,
                    }
                )

        return "I see."

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        scenario = self.session_state.get(session_id, metadata)
        agent_text = extract_text(action)

        user_reply = await self._get_user_reply(agent_text, scenario)

        if "resolves my issue" in user_reply.lower():
            keywords = scenario.get("resolution_keywords", [])
            hit = any(kw.lower() in agent_text.lower() for kw in keywords)
            return None, 1.0 if hit else 0.5, True, False, {"resolved": True, "keyword_hit": hit}

        return user_reply, 0.0, False, False, {}


if __name__ == "__main__":
    CustomerServiceEnv.run_webserver()
