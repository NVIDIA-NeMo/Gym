from typing import Annotated, TypedDict

from app import LangGraphAgentAdapter, LangGraphAgentConfig
from fastapi import Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


class ReflectionAgentConfig(LangGraphAgentConfig):
    max_reflections: int = 2


class ReflectionAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ReflectionAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ReflectionAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ReflectionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    nemo_outputs: list
    cookies: dict
    reflections: int
    request_body: NeMoGymResponseCreateParamsNonStreaming
    last_model_response: NeMoGymResponse


class ReflectionAgent(LangGraphAgentAdapter):
    config: ReflectionAgentConfig

    def build_graph(self):
        graph = StateGraph(ReflectionState)

        async def generate(state):
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            input_messages = [
                NeMoGymEasyInputMessage(role=role_map.get(m.type, m.type), content=m.content)
                for m in state["messages"]
            ]

            request_body = state["request_body"].model_copy(update={"input": input_messages + state["nemo_outputs"]})

            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=request_body,
                cookies=state["cookies"],
            )

            await raise_for_status(resp)
            nemo_response = NeMoGymResponse.model_validate(await resp.json())

            new_outputs = nemo_response.output
            all_outputs = state["nemo_outputs"] + new_outputs

            text = "".join(
                c.text for o in new_outputs if o.type == "message" for c in o.content if c.type == "output_text"
            )

            return {
                "messages": [AIMessage(content=text)],
                "nemo_outputs": all_outputs,
                "cookies": resp.cookies,
                "reflections": state["reflections"],
                "last_model_response": nemo_response,
                "request_body": state["request_body"],
            }

        async def reflect(state):
            reflection_prompt = NeMoGymEasyInputMessage(
                role="user", content="Critique your solution. What could be wrong?"
            )

            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            input_messages = [
                NeMoGymEasyInputMessage(role=role_map.get(m.type, m.type), content=m.content)
                for m in state["messages"]
            ] + [reflection_prompt]

            request_body = state["request_body"].model_copy(update={"input": input_messages + state["nemo_outputs"]})

            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=request_body,
                cookies=state["cookies"],
            )

            await raise_for_status(resp)
            nemo_response = NeMoGymResponse.model_validate(await resp.json())

            text = "".join(
                c.text
                for o in nemo_response.output
                if o.type == "message"
                for c in o.content
                if c.type == "output_text"
            )

            return {
                "messages": [
                    HumanMessage(content="Critique your solution. What could be wrong?"),
                    AIMessage(content=text),
                ],
                "nemo_outputs": state["nemo_outputs"] + [reflection_prompt] + nemo_response.output,
                "cookies": resp.cookies,
                "reflections": state["reflections"] + 1,
                "last_model_response": nemo_response,
                "request_body": state["request_body"],
            }

        def should_continue(state):
            if state["reflections"] >= self.config.max_reflections:
                return END
            last = state["messages"][-1].content if state["messages"] else ""
            return END if "<answer>" in last else "reflect"

        graph.add_node("generate", generate)
        graph.add_node("reflect", reflect)
        graph.set_entry_point("generate")
        graph.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
        graph.add_edge("reflect", "generate")

        return graph.compile()

    async def get_initial_state(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict) -> dict:
        if isinstance(body.input, str):
            initial_messages = [HumanMessage(content=body.input)]
            nemo_outputs = []
        else:
            initial_messages = []
            nemo_outputs = []
            for msg in body.input:
                is_output = (
                    hasattr(msg, "type")
                    and msg.type == "message"
                    and hasattr(msg, "role")
                    and msg.role == "assistant"
                    and hasattr(msg, "content")
                    and isinstance(msg.content, list)
                )
                is_function_call = hasattr(msg, "type") and msg.type == "function_call"

                if is_output or is_function_call:
                    nemo_outputs.append(msg)
                else:
                    role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                    if role in ["user", "human"]:
                        initial_messages.append(HumanMessage(content=content))
                    elif role in ["assistant", "ai"]:
                        initial_messages.append(AIMessage(content=content))

        return {
            "messages": initial_messages,
            "nemo_outputs": nemo_outputs,
            "cookies": cookies,
            "reflections": 0,
            "request_body": body,
            "last_model_response": None,
        }

    def extract_outputs(self, final_state: dict) -> list:
        return final_state["nemo_outputs"]

    async def run(self, request: Request, body: ReflectionAgentRunRequest) -> ReflectionAgentVerifyResponse:
        cookies = request.cookies

        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        resp = await self.server_client.post(
            server_name=self.config.name, url_path="/v1/responses", json=body.responses_create_params, cookies=cookies
        )
        await raise_for_status(resp)

        verify_request = ReflectionAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(resp)}
        )

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resp.cookies,
        )
        await raise_for_status(verify)
        return ReflectionAgentVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    ReflectionAgent.run_webserver()
