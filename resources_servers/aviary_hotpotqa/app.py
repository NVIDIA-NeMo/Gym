from typing import Dict, Optional
from pydantic import BaseModel, PrivateAttr
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

class AviaryHotpotqaResourcesServerConfig(BaseResourcesServerConfig):
    max_steps: int = 10
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0

class SearchRequest(BaseModel):
    entity: str
    question: Optional[str] = None
    ground_truth: Optional[str] = None

class SearchResponse(BaseModel):
    result: str
    session_id: str

class LookupRequest(BaseModel):
    keyword: str

class LookupResponse(BaseModel):
    result: str
    session_id: str

class SubmitAnswerRequest(BaseModel):
    answer: str
    question: Optional[str] = None
    ground_truth: Optional[str] = None

class SubmitAnswerResponse(BaseModel):
    result: str
    reward: float
    done: bool
    session_id: str

class AviaryHotpotqaRunRequest(BaseRunRequest):
    session_id: Optional[str] = None

class AviaryHotpotqaVerifyRequest(AviaryHotpotqaRunRequest, BaseVerifyRequest):
    pass

class AviaryHotpotqaVerifyResponse(BaseVerifyResponse):
    pass

class AviaryHotpotqaResourcesServer(SimpleResourcesServer):
    config: AviaryHotpotqaResourcesServerConfig
    
    _sessions: Dict[str, dict] = PrivateAttr(default_factory=dict) # cant serialize aviary env tools, i think, but dont think this is needed... Field might be sufficient since its stateful so no serialization 
    
    def __init__(self, **data):
        super().__init__(**data)

    def _create_env_config(self, question: str, ground_truth: str) -> dict:
        return {
            "question_id": None,
            "question": question,
            "correct_answer": ground_truth,
            "correct_reward": self.config.correct_reward,
            "incorrect_reward": self.config.incorrect_reward,
            "tool_failure_reward": 0.0,
        }

    def _create_session(self, env: HotPotQAEnv, env_config: dict) -> dict:
        return {
            "env": env,
            "env_config": env_config,
            "turn_count": 0,
            "tracking": {},
            "reward": 0.0
        }

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
        
        try:
            session = self._sessions[session_id]
            env = session["env"]
            
            tool_call = ToolCall.from_name("search", entity=body.entity)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            session["turn_count"] += 1
            session["reward"] = float(reward)
            session["last_done"] = done
            result = self._extract_message_content(msgs)
            
            return SearchResponse(result=result, session_id=session_id)
            
        except Exception as e:
            return SearchResponse(result=f"Error during search: {str(e)}", session_id=session_id)
    
    async def lookup(self, body: LookupRequest, request: Request) -> LookupResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        if session_id not in self._sessions:
            return LookupResponse(result="No active search session. Please search first.", session_id=session_id)
        
        try:
            session = self._sessions[session_id]
            env = session["env"]
            
            tool_call = ToolCall.from_name("lookup", keyword=body.keyword)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            session["turn_count"] += 1
            session["reward"] = float(reward)
            session["last_done"] = done
            result = self._extract_message_content(msgs)
            
            return LookupResponse(result=result, session_id=session_id)
            
        except Exception as e:
            return LookupResponse(result=f"Error during lookup: {str(e)}", session_id=session_id)
    
    async def submit_answer(self, body: SubmitAnswerRequest, request: Request) -> SubmitAnswerResponse:
        session_id = request.session[SESSION_ID_KEY]
        
        print(f"DEBUG SUBMIT: session_id={session_id}")
        print(f"DEBUG SUBMIT: body.answer={body.answer}")
        print(f"DEBUG SUBMIT: body.question={body.question}")
        print(f"DEBUG SUBMIT: body.ground_truth={body.ground_truth}")
        
        await self._init_env(session_id, body.question, body.ground_truth)
        
        try:
            session = self._sessions[session_id]
            env = session["env"]
            
            tool_call = ToolCall.from_name("submit_answer", answer=body.answer)
            action = ToolRequestMessage(tool_calls=[tool_call])
            msgs, reward, done, truncated = await env.step(action)
            
            print(f"DEBUG SUBMIT: reward={reward}")
            
            session["reward"] = float(reward)
            session["last_done"] = done
            result = self._extract_message_content(msgs)
            
            return SubmitAnswerResponse(
                result=result,
                reward=float(reward),
                done=done,
                session_id=session_id
            )
            
        except Exception as e:
            return SubmitAnswerResponse(
                result=f"Error submitting answer: {str(e)}",
                reward=0.0,
                done=True,
                session_id=session_id
            )

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """returns reward from the session state"""
        session_id = request.session[SESSION_ID_KEY]
        
        reward = 0.0
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            reward = session.get("reward", 0.0)
            
        print(f"DEBUG VERIFY: session_id={session_id}")
        print(f"DEBUG VERIFY: session exists={session_id in self._sessions}")
        
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            env_config = session.get("env_config", {})
            
            print(f"DEBUG VERIFY: question={env_config.get('question')}")
            print(f"DEBUG VERIFY: correct_answer={env_config.get('correct_answer')}")
            print(f"DEBUG VERIFY: reward={reward}")
        
        # should cleanup? but ran into errors initially    
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    AviaryHotpotqaResourcesServer.run_webserver()
