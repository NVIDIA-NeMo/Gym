"""Mock OpenAI server that serves pre-baked answers from a local JSONL database.

Each entry in mock_responses.jsonl has a key (question prefix), expected_answer,
and candidate answers with varying correctness. The server matches prompts by
question prefix and randomly samples from candidates.

Usage: python tests/mock_openai_server.py  (port 8111)
"""

import json
import random
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

# Load response database: list of (key, candidates) sorted by key length descending for greedy match
DB: list[tuple[str, list[str]]] = []
db_path = Path(__file__).parent / "mock_responses.jsonl"
if db_path.exists():
    with open(db_path) as f:
        for line in f:
            entry = json.loads(line)
            DB.append((entry["key"], entry["candidates"]))
    DB.sort(key=lambda x: len(x[0]), reverse=True)

THINK_TEMPLATES = [
    "<think>Let me analyze this step by step. After working through the algebra, I find the answer.</think>\n\n",
    "<think>Breaking this down carefully. Considering the constraints, I solve systematically.</think>\n\n",
    "<think>This requires careful reasoning. Applying the formulas and simplifying.</think>\n\n",
]


def _find_answer(text: str) -> str | None:
    """Returns an answer string, or None for garbled (no extraction possible)."""
    for key, candidates in DB:
        if key in text:
            return random.choice(candidates)
    return str(random.randint(1, 999))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    text = " ".join(m.get("content", "") for m in body.get("messages", []))
    answer = _find_answer(text)
    if answer is None:
        # Garbled response — no boxed answer, simulates extraction failure
        content = f"{random.choice(THINK_TEMPLATES)}I believe the answer involves complex analysis but I'm not fully certain of the exact value."
    else:
        content = f"{random.choice(THINK_TEMPLATES)}The answer is $\\boxed{{{answer}}}$."
    pt = random.randint(80, 150)
    ct = random.randint(200, 800)
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
    )


@app.post("/tokenize")
async def tokenize(request: Request):
    body = await request.json()
    n = max(1, len(body.get("prompt", "") or body.get("text", "")) // 4)
    return JSONResponse({"tokens": list(range(n)), "count": n})


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "owned_by": "test",
                    "created": int(time.time()),
                    "max_model_len": 32768,
                }
            ],
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
