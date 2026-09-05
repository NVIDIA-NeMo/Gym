# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-state meal-planning example for a user simulator and assistant."""

from typing import Any, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.processors.user_assistant import UserAssistantVerifyRequest, UserAssistantVerifyResponse
from nemo_gym.server_utils import SESSION_ID_KEY


class MealRecommendation(BaseModel):
    name: str
    diet: str
    price: float = Field(gt=0)


class SupportState(BaseModel):
    preferences: dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[MealRecommendation] = None
    accepted: bool = False


class ExampleUserAssistantConfig(BaseResourcesServerConfig):
    pass


class ExampleUserAssistantSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class SavePreferenceRequest(BaseModel):
    diet: str
    max_price: float = Field(gt=0)


class SavePreferenceResponse(BaseModel):
    saved: bool
    preferences: dict[str, Any]


class ReadPreferencesResponse(BaseModel):
    preferences: dict[str, Any]


class RecommendMealRequest(MealRecommendation):
    pass


class RecommendMealResponse(BaseModel):
    saved: bool
    recommendation: MealRecommendation


class AcceptRecommendationResponse(BaseModel):
    accepted: bool
    recommendation: Optional[MealRecommendation]


class EpisodeStatusResponse(BaseModel):
    terminated: bool
    reason: Optional[str] = None
    state: dict[str, Any]


class ExampleUserAssistantVerifyResponse(UserAssistantVerifyResponse):
    preference_satisfied: bool


class ExampleUserAssistantServer(SimpleResourcesServer):
    config: ExampleUserAssistantConfig
    session_id_to_state: dict[str, SupportState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/save_preference")(self.save_preference)
        app.post("/read_preferences")(self.read_preferences)
        app.post("/recommend_meal")(self.recommend_meal)
        app.post("/accept_recommendation")(self.accept_recommendation)
        app.post("/episode_status")(self.episode_status)
        return app

    def _state(self, request: Request) -> SupportState:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_state:
            raise RuntimeError("No active session. Call /seed_session first.")
        return self.session_id_to_state[session_id]

    async def seed_session(
        self,
        request: Request,
        body: ExampleUserAssistantSeedRequest,
    ) -> BaseSeedSessionResponse:
        del body
        self.session_id_to_state[request.session[SESSION_ID_KEY]] = SupportState()
        return BaseSeedSessionResponse()

    async def save_preference(self, request: Request, body: SavePreferenceRequest) -> SavePreferenceResponse:
        state = self._state(request)
        state.preferences = {"diet": body.diet.lower(), "max_price": body.max_price}
        return SavePreferenceResponse(saved=True, preferences=state.preferences)

    async def read_preferences(self, request: Request) -> ReadPreferencesResponse:
        return ReadPreferencesResponse(preferences=self._state(request).preferences)

    async def recommend_meal(self, request: Request, body: RecommendMealRequest) -> RecommendMealResponse:
        state = self._state(request)
        state.recommendation = MealRecommendation.model_validate(body)
        return RecommendMealResponse(saved=True, recommendation=state.recommendation)

    async def accept_recommendation(self, request: Request) -> AcceptRecommendationResponse:
        state = self._state(request)
        state.accepted = state.recommendation is not None
        return AcceptRecommendationResponse(accepted=state.accepted, recommendation=state.recommendation)

    async def episode_status(self, request: Request) -> EpisodeStatusResponse:
        state = self._state(request)
        return EpisodeStatusResponse(
            terminated=state.accepted,
            reason="recommendation_accepted" if state.accepted else None,
            state=state.model_dump(mode="json"),
        )

    async def verify(
        self,
        request: Request,
        body: UserAssistantVerifyRequest,
    ) -> ExampleUserAssistantVerifyResponse:
        state = self._state(request)
        recommendation = state.recommendation
        preference_satisfied = bool(
            state.accepted
            and recommendation is not None
            and recommendation.diet.lower() == state.preferences.get("diet")
            and recommendation.price <= state.preferences.get("max_price", 0)
        )
        return ExampleUserAssistantVerifyResponse(
            **body.model_dump(mode="json"),
            reward=float(preference_satisfied),
            preference_satisfied=preference_satisfied,
        )


if __name__ == "__main__":
    ExampleUserAssistantServer.run_webserver()
