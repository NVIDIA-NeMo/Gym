from typing import Dict, Optional
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
from fastapi import FastAPI, Request

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.server_utils import SESSION_ID_KEY

from aviary.envs.hotpotqa.env import HotPotQAEnv
from aviary.core import ToolCall, ToolRequestMessage

class AviarySession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    env: HotPotQAEnv
    env_config: dict
    turn_count: int = 0
    
    @property
    def reward(self) -> float:
        return self.env.state.reward
    
    @property 
    def done(self) -> bool:
        return self.env.state.done
    
    @property
    def steps(self) -> int:
        return self.env.state.steps
        
    @property
    def answer(self) -> str | None:
        return self.env.state.answer

class AviaryHotpotqaResourcesServerConfig(BaseResourcesServerConfig):
    max_steps: int = 10

class SearchRequest(BaseModel):
    """
    Maps to aviary's search tool parameters:
    - entity: The Wikipedia entity to search for
    
    See: https://github.com/Future-House/aviary/blob/main/packages/hotpotqa/src/aviary/envs/hotpotqa/env.py#L401
    """
    entity: str
    question: Optional[str] = None
    ground_truth: Optional[str] = None

class SearchResponse(BaseModel):
    """
    Based on aviary.tools.base.ToolResponseMessage:
    - result: Equivalent to ToolResponseMessage.content
    
    See: https://github.com/Future-House/aviary/blob/main/src/aviary/tools/base.py#L150
    """
    result: str

class LookupRequest(BaseModel):
    """
    Maps to aviary's lookup tool parameters:
    - keyword: The keyword to search for in the current page
    
    See: https://github.com/Future-House/aviary/blob/main/packages/hotpotqa/src/aviary/envs/hotpotqa/env.py#L469
    """
    keyword: str

class LookupResponse(BaseModel):
    """
    Based on aviary.tools.base.ToolResponseMessage:
    - result: Equivalent to ToolResponseMessage.content
    
    See: https://github.com/Future-House/aviary/blob/main/src/aviary/tools/base.py#L150
    """
    result: str

class SubmitAnswerRequest(BaseModel):
    """
    Maps to aviary's submit_answer tool parameters:
    - answer: The final answer to submit
    
    See: https://github.com/Future-House/aviary/blob/main/packages/hotpotqa/src/aviary/envs/hotpotqa/env.py#L385
    """
    answer: str
    question: Optional[str] = None
    ground_truth: Optional[str] = None

class SubmitAnswerResponse(BaseModel):
    """
    Based on:
    - aviary.tools.base.ToolResponseMessage (result -> content)
    - aviary.envs.hotpotqa.env.HotPotQAEnvState (reward, done)
    
    See: 
    - https://github.com/Future-House/aviary/blob/main/src/aviary/tools/base.py#L150
    - https://github.com/Future-House/aviary/blob/main/packages/hotpotqa/src/aviary/envs/hotpotqa/env.py#L61d
    """
    result: str
    reward: float
    done: bool

class AviaryHotpotqaRunRequest(BaseRunRequest):
    pass

class AviaryHotpotqaVerifyRequest(AviaryHotpotqaRunRequest, BaseVerifyRequest):
    pass

class AviaryHotpotqaVerifyResponse(BaseVerifyResponse):
    pass

class AviaryHotpotqaResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AviaryHotpotqaResourcesServerConfig
    _sessions: Dict[str, AviarySession] = PrivateAttr(default_factory=dict) 
    
    def __init__(self, **data):
        super().__init__(**data)

    def _create_env_config(self, question: str, ground_truth: str) -> dict:
        # aviary defaults: correct_reward=1.0, incorrect_reward=0.0, tool_failure_reward=0.0
        return {
            "question_id": None,
            "question": question,
            "correct_answer": ground_truth,
        }

    def _create_session(self, env: HotPotQAEnv, env_config: dict) -> AviarySession:
        return AviarySession(
            env=env,
            env_config=env_config,
            turn_count=0
        )

    async def _init_env(self, session_id: str, question: Optional[str], ground_truth: Optional[str]) -> None:
        """Init env and session if doesn't exist yet (first tool call)"""
        if session_id not in self._sessions:
            if not question or not ground_truth:
                raise ValueError("Question or ground truth not found in tool payload")
            
            env_config = self._create_env_config(question, ground_truth)
            env = HotPotQAEnv(**env_config)
            await env.reset()
            
            self._sessions[session_id] = self._create_session(env, env_config)

    def _extract_message_content(self, msgs) -> str:
        if msgs and len(msgs) > 0:
            return msgs[0].content
        return "No response from environment"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/search")(self.search)
        app.post("/lookup")(self.lookup)
        app.post("/submit_answer")(self.submit_answer)

        return app


    async def search(self, body: SearchRequest, request: Request) -> SearchResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        await self._init_env(session_id, body.question, body.ground_truth)
        session = self._sessions[session_id]
        env = session.env
        
        try:
            tool_call = ToolCall.from_name("search", entity=body.entity)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            session.turn_count += 1
            result = self._extract_message_content(msgs)
            
            return SearchResponse(result=result)
            
        except Exception as e:
            return SearchResponse(result=f"Error during search: {str(e)}")
    
    async def lookup(self, body: LookupRequest, request: Request) -> LookupResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        if session_id not in self._sessions:
            return LookupResponse(result="No active search session. Please search first.")
        
        session = self._sessions[session_id]
        env = session.env
        
        try:
            tool_call = ToolCall.from_name("lookup", keyword=body.keyword)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            session.turn_count += 1
            result = self._extract_message_content(msgs)
            
            return LookupResponse(result=result)
            
        except Exception as e:
            return LookupResponse(result=f"Error during lookup: {str(e)}")
    
    async def submit_answer(self, body: SubmitAnswerRequest, request: Request) -> SubmitAnswerResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        print(f"DEBUG SUBMIT: session_id={session_id}")
        print(f"DEBUG SUBMIT: body.answer={body.answer}")
        print(f"DEBUG SUBMIT: body.question={body.question}")
        print(f"DEBUG SUBMIT: body.ground_truth={body.ground_truth}")
        
        await self._init_env(session_id, body.question, body.ground_truth)
        session = self._sessions[session_id]
        env = session.env
        
        try:
            tool_call = ToolCall.from_name("submit_answer", answer=body.answer)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            print(f"DEBUG SUBMIT: reward={reward}")
            
            result = self._extract_message_content(msgs)
            
            return SubmitAnswerResponse(
                result=result,
                reward=session.reward,
                done=session.done,
            )
            
        except Exception as e:
            return SubmitAnswerResponse(
                result=f"Error submitting answer: {str(e)}",
                reward=0.0,
                done=True,
            )

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """returns reward from the session state"""
        session_id = request.session[SESSION_ID_KEY]
        
        reward = 0.0
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            reward = session.reward
            
        print(f"DEBUG VERIFY: session_id={session_id}")
        print(f"DEBUG VERIFY: session exists={session_id in self._sessions}")
        
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            env_config = session.env_config
            
            print(f"DEBUG VERIFY: question={env_config.get('question')}")
            print(f"DEBUG VERIFY: correct_answer={env_config.get('correct_answer')}")
            print(f"DEBUG VERIFY: reward={reward}")
        
        # should cleanup? but ran into errors initially    
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

if __name__ == "__main__":
    AviaryHotpotqaResourcesServer.run_webserver()