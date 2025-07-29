from httpx import Limits, AsyncClient, AsyncHTTPTransport

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses import Response


class NeMoGymResponseCreateParamsNonStreaming(ResponseCreateParamsNonStreaming):
    pass


class NeMoGymResponse(Response):
    pass


class NeMoGymAsyncOpenAI(AsyncOpenAI):
    def __init__(self, **kwargs) -> None:
        # We override the http_client and the timeout here to use the maximum reasonably possible.
        # TODO: this setup is take from https://github.com/NVIDIA/NeMo-Skills/blob/80dc78ac758c4cac81c83a43a729e7ca1280857b/nemo_skills/inference/model/base.py#L318
        # However, there is still a lingering issue regarding saturating at 100 max connections
        http_client = AsyncClient(
            limits=Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=AsyncHTTPTransport(retries=3),
        )
        kwargs["http_client"] = http_client

        kwargs["timeout"] = None  # Enforce no timeout

        super().__init__(**kwargs)
