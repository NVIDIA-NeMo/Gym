# NeMo Gym Fix Patterns

Correct implementations for each anti-pattern. These are production code patterns — use them directly.

---

## aiohttp-adapter

When wrapping an external library that uses httpx internally, replace its HTTP transport with an aiohttp-compatible adapter:

```python
from pydantic import BaseModel
from nemo_gym.server_utils import request, raise_for_status

class AIOHTTPClientResponse(BaseModel):
    """Drop-in replacement for httpx.Response."""
    status_code: int
    data: dict

    def json(self):
        return self.data


class AIOHTTPClient(BaseModel):
    """Drop-in replacement for httpx.AsyncClient.
    
    Wraps aiohttp (via nemo_gym.server_utils.request) to avoid
    httpx's O(n^2) connection pooling at high concurrency.
    """
    headers: dict
    base_url: str

    async def post(self, endpoint: str, content: str, timeout: float) -> AIOHTTPClientResponse:
        response = await request(
            method="POST",
            headers=self.headers,
            url=f"{self.base_url}{endpoint}",
            data=content,
        )
        return AIOHTTPClientResponse(
            status_code=response.status,
            data=await response.json(),
        )

    @classmethod
    def from_httpx_client(cls, client, **kwargs):
        """Convert an existing httpx.AsyncClient to this adapter."""
        return cls(
            headers=dict(client.headers),
            base_url=str(client.base_url),
            **kwargs,
        )
```

Usage in `model_post_init()`:
```python
def model_post_init(self, __context):
    super().model_post_init(__context)
    # Replace the library's httpx client with aiohttp adapter
    self.library._client = AIOHTTPClient.from_httpx_client(self.library._client)
```

---

## cookie-propagation

Full cookie chain for a multi-turn agent:

```python
async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
    cookies = request.cookies

    # Seed session
    seed_resp = await self.server_client.post(
        server_name=self.config.resources_server.name,
        url_path="/seed_session",
        json=body.model_dump(),
        cookies=cookies,
    )
    await raise_for_status(seed_resp)
    cookies = seed_resp.cookies  # Update cookies from response

    for turn in range(self.config.max_turns):
        # Model call
        gen_resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=current_input,
            cookies=cookies,  # Forward cookies
        )
        await raise_for_status(gen_resp)
        cookies = gen_resp.cookies  # Update

        # Verify call
        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_data,
            cookies=cookies,  # Forward cookies
        )
        await raise_for_status(verify_resp)
        cookies = verify_resp.cookies  # Update
```

---

## token-id-propagation

Accumulate token IDs across all turns in a multi-turn agent:

```python
all_prompt_token_ids = []
all_generation_token_ids = []
all_generation_log_probs = []

for turn in range(max_turns):
    model_response = await get_response_json(gen_resp)

    # Accumulate from each model call
    all_prompt_token_ids.extend(model_response.get("prompt_token_ids", []))
    all_generation_token_ids.extend(model_response.get("generation_token_ids", []))
    all_generation_log_probs.extend(model_response.get("generation_log_probs", []))

    # ... verify, check reward, build next turn ...

# Attach to final response
final_response.prompt_token_ids = all_prompt_token_ids
final_response.generation_token_ids = all_generation_token_ids
final_response.generation_log_probs = all_generation_log_probs
```

---

## semaphore-subprocess

Bound concurrent subprocess execution:

```python
class MyServer(SimpleResourcesServer):
    config: MyConfig

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.semaphore = asyncio.Semaphore(self.config.num_processes)

    async def verify(self, body):
        async with self.semaphore:
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )
            output = stdout.decode(errors="replace")
            errors = stderr.decode(errors="replace")
```

---

## think-block-stripping

Three patterns depending on context:

**Simple strip (most common):**
```python
if "</think>" in text:
    text = text.split("</think>")[-1].strip()
```

**Violation detection (for RL penalty):**
```python
def has_reasoning_format_violation(response) -> bool:
    final_answer = response.output_text or ""
    if "<think>" in final_answer or "</think>" in final_answer:
        return True
    # Check reasoning content for duplicate tags
    reasoning = extract_reasoning_text(response)
    if reasoning.count("<think>") > 1 or reasoning.count("</think>") > 1:
        return True
    return False
```

**Structured parsing (for multi-section output):**
```python
response = response.split("</think>")[-1].strip()
if SOLUTION_HEADER not in response:
    return None, "missing_solution_header"
proof, self_eval = response.split(SELF_EVAL_HEADER, 1)
```
