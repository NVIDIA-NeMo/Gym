from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# --- Chat Completions API ---
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=500,
    temperature=0.7,
)
print("Response:", response.choices[0].message.content)
