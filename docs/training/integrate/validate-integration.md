---
description: "Verify your training framework integration works correctly end-to-end"
categories: ["how-to-guides"]
tags: ["validation", "testing", "integration", "troubleshooting", "debugging"]
personas: ["mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
---

(integrate-validate-integration)=

# Validate Your Integration

Run a validation suite to verify your training framework integration works correctly end-to-end.

## How It Works

The validation suite tests each layer of the integration:

1. **HTTP endpoint** — Verify the OpenAI-compatible server responds correctly
2. **Token stability** — Ensure tokenization round-trips are consistent
3. **Rollout collection** — Confirm Gym returns complete rollouts with rewards
4. **Token continuity** — Validate token sequences align across turns
5. **Gradient flow** — Test that training data produces valid gradients

## Before You Start

**Prerequisites**:

- Completed {doc}`expose-openai-endpoint` and {doc}`connect-gym-to-training`
- Working rollout collection
- Access to your tokenizer and model (for gradient tests)

---

## Validation Checklist

```{list-table}
:header-rows: 1
:widths: 30 50 20

* - Test
  - What It Validates
  - Required
* - HTTP Endpoint
  - `/v1/models` and `/v1/chat/completions` respond
  - Yes
* - Token Round-Trip
  - Encode → decode → re-encode produces same IDs
  - Yes
* - Rollout Collection
  - Gym returns rollouts with rewards and token data
  - Yes
* - Token Continuity
  - Token sequences are contiguous across turns
  - Yes
* - Gradient Flow
  - Processed rollouts produce valid gradients
  - Optional
```

---

## Run Validation

### Test HTTP Endpoint

Verify your OpenAI-compatible endpoint handles required operations:

```bash
# Quick test
curl http://localhost:8000/v1/models
```

:::{dropdown} Full Endpoint Test (Python)
:icon: code

```python
import aiohttp
import asyncio


async def test_endpoint(base_url: str):
    """Test HTTP endpoint compatibility."""
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Model listing
        async with session.get(f"{base_url}/v1/models") as resp:
            assert resp.status == 200, f"Models endpoint failed: {resp.status}"
            data = await resp.json()
            assert "data" in data, "No models returned"
            print("✅ Model listing works")
        
        # Test 2: Chat completions
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": data["data"][0]["id"],
                "messages": [{"role": "user", "content": "Say 'test'"}],
                "max_tokens": 10,
            }
        ) as resp:
            assert resp.status == 200, f"Chat completions failed: {resp.status}"
            result = await resp.json()
            assert "choices" in result, "No choices in response"
            print("✅ Chat completions work")
        
        # Test 3: Tokenization (optional)
        async with session.post(
            f"{base_url}/tokenize",
            json={"prompt": "Hello world"}
        ) as resp:
            if resp.status == 200:
                print("✅ Tokenization endpoint works")
            else:
                print("⚠️  Tokenization endpoint not available (optional)")


asyncio.run(test_endpoint("http://localhost:8000"))
```

:::

### Test Token Round-Trip Stability

Verify tokenization is stable across encode/decode cycles:

:::{dropdown} Token Round-Trip Test
:icon: code

```python
def test_token_roundtrip(tokenizer, test_strings: list[str]):
    """Test that tokenization round-trips are stable."""
    failures = []
    
    for text in test_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)
        
        if token_ids != re_encoded:
            failures.append({
                "original": text,
                "token_ids": token_ids,
                "decoded": decoded,
                "re_encoded": re_encoded,
            })
    
    if failures:
        print(f"❌ {len(failures)} round-trip failures:")
        for f in failures[:3]:
            print(f"   '{f['original']}' -> {f['token_ids']} -> '{f['decoded']}' -> {f['re_encoded']}")
        return False
    
    print(f"✅ All {len(test_strings)} round-trips stable")
    return True


test_strings = [
    "Hello, world!",
    "The answer is 42.",
    "```python\nprint('hello')\n```",
    "Multi\nline\ntext",
]
test_token_roundtrip(tokenizer, test_strings)
```

:::

### Test End-to-End Rollout

Collect a rollout and verify all required fields:

:::{dropdown} Rollout Collection Test
:icon: code

```python
async def test_rollout_collection(gym: GymIntegration):
    """Test complete rollout collection."""
    
    example = {
        "responses_create_params": {
            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
            "model": "policy",
        },
        "agent_ref": "your_agent_name",
    }
    
    async for rollout in gym.collect_rollouts([example]):
        # Verify structure
        assert "response" in rollout, "Missing response"
        assert "output" in rollout["response"], "Missing output"
        assert "reward" in rollout, "Missing reward"
        assert 0.0 <= rollout["reward"] <= 1.0, f"Invalid reward: {rollout['reward']}"
        
        # Check for trainable outputs
        trainable_count = sum(
            1 for o in rollout["response"]["output"]
            if "generation_token_ids" in o
        )
        assert trainable_count > 0, "No trainable outputs in rollout"
        
        print(f"✅ Rollout collected successfully")
        print(f"   Reward: {rollout['reward']}")
        print(f"   Trainable outputs: {trainable_count}")
        
        return rollout
```

:::

### Test Token Continuity

Verify token sequences align correctly across turns:

:::{dropdown} Token Continuity Test
:icon: code

```python
def test_token_continuity(rollout: dict, tokenizer):
    """Test that token sequences are contiguous."""
    
    trainable_turns = extract_trainable_turns(rollout)
    
    try:
        validate_token_continuity(trainable_turns)
        print(f"✅ Token continuity validated across {len(trainable_turns)} turns")
        return True
    except AssertionError as e:
        print(f"❌ Token continuity failed: {e}")
        return False
```

:::

### Test Gradient Flow (Optional)

Verify processed rollouts produce valid gradients:

:::{dropdown} Gradient Flow Test
:icon: code

```python
import torch


def test_gradient_flow(processed_rollout: dict, model: torch.nn.Module):
    """Test that gradients flow correctly through processed data."""
    
    trainable_ids = []
    for msg in processed_rollout["message_log"]:
        if msg.get("trainable", False):
            trainable_ids.extend(msg["token_ids"].tolist())
    
    if not trainable_ids:
        print("⚠️  No trainable tokens in rollout")
        return False
    
    input_ids = torch.tensor([trainable_ids])
    
    model.train()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    
    if has_gradients:
        print(f"✅ Gradients flow correctly (loss={loss.item():.4f})")
        return True
    else:
        print("❌ No gradients computed")
        return False
```

:::

---

## Full Validation Suite

Run all tests together:

:::{dropdown} Complete Validation Script
:icon: play

```python
async def run_validation_suite(
    base_url: str,
    gym: GymIntegration,
    tokenizer,
    model=None,
):
    """Run complete validation suite."""
    
    print("=" * 50)
    print("Training Framework Integration Validation")
    print("=" * 50)
    
    results = {}
    
    # Test 1: HTTP Endpoint
    print("\n📡 Testing HTTP endpoint...")
    try:
        await test_endpoint(base_url)
        results["http_endpoint"] = "✅ PASS"
    except Exception as e:
        results["http_endpoint"] = f"❌ FAIL: {e}"
    
    # Test 2: Token Round-Trip
    print("\n🔄 Testing token round-trips...")
    test_strings = ["Hello!", "The answer is 42", "```code```"]
    if test_token_roundtrip(tokenizer, test_strings):
        results["token_roundtrip"] = "✅ PASS"
    else:
        results["token_roundtrip"] = "❌ FAIL"
    
    # Test 3: Rollout Collection
    print("\n📦 Testing rollout collection...")
    try:
        rollout = await test_rollout_collection(gym)
        results["rollout_collection"] = "✅ PASS"
    except Exception as e:
        results["rollout_collection"] = f"❌ FAIL: {e}"
        rollout = None
    
    # Test 4: Token Continuity
    if rollout:
        print("\n🔗 Testing token continuity...")
        if test_token_continuity(rollout, tokenizer):
            results["token_continuity"] = "✅ PASS"
        else:
            results["token_continuity"] = "❌ FAIL"
        
        # Test 5: Gradient Flow (optional)
        if model:
            print("\n📈 Testing gradient flow...")
            processed = process_rollout_for_training(rollout, tokenizer)
            if processed and test_gradient_flow(processed, model):
                results["gradient_flow"] = "✅ PASS"
            else:
                results["gradient_flow"] = "❌ FAIL"
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test, result in results.items():
        print(f"  {test}: {result}")
        if "FAIL" in result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validations passed! Your integration is ready.")
    else:
        print("\n⚠️  Some validations failed. Review the errors above.")
    
    return all_passed


# Run validation
# asyncio.run(run_validation_suite(base_url, gym, tokenizer, model))
```

:::

---

## Expected Output

A successful validation produces:

```text
==================================================
Training Framework Integration Validation
==================================================

📡 Testing HTTP endpoint...
✅ Model listing works
✅ Chat completions work

🔄 Testing token round-trips...
✅ All 3 round-trips stable

📦 Testing rollout collection...
✅ Rollout collected successfully
   Reward: 1.0
   Trainable outputs: 2

🔗 Testing token continuity...
✅ Token continuity validated across 2 turns

==================================================
VALIDATION SUMMARY
==================================================
  http_endpoint: ✅ PASS
  token_roundtrip: ✅ PASS
  rollout_collection: ✅ PASS
  token_continuity: ✅ PASS

🎉 All validations passed! Your integration is ready.
```

---

## Troubleshooting

:::{dropdown} HTTP Endpoint Tests Fail
:icon: alert

**Solutions**:
- Refer to {doc}`expose-openai-endpoint`
- Verify vLLM configuration has `expose_http_server: true`
- Check that the server has fully initialized

:::

:::{dropdown} Token Round-Trips Unstable
:icon: alert

Some tokenizers behave inconsistently.

**Solutions**:
- Use `add_special_tokens=False` consistently
- Strip whitespace before comparison
- File an issue with tokenizer maintainers if persistent

:::

:::{dropdown} Rollout Collection Fails
:icon: alert

**Solutions**:
- Refer to {doc}`connect-gym-to-training`
- Verify Gym servers are running
- Check agent configuration is correct
- Test network connectivity between components

:::

:::{dropdown} Token Continuity Fails
:icon: alert

**Solutions**:
- Refer to {doc}`/training/rollout-collection/process-multi-turn-rollouts`
- Check that trainable turns have correct `prompt_token_ids`
- Verify no tokens are missing between turns

:::

---

## Integration Complete

Your training framework integration is validated and ready for production use.

**Completed steps**:

1. ✅ {doc}`expose-openai-endpoint` — HTTP endpoint configured
2. ✅ {doc}`connect-gym-to-training` — Gym integrated into training loop
3. ✅ {doc}`validate-integration` — Integration validated

:::{tip}
For multi-turn agentic tasks, refer to {doc}`/training/rollout-collection/process-multi-turn-rollouts` to handle token alignment across turns.
:::

---

## Next Steps

- **Scale up training** — Refer to {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl` to see how NeMo RL scales Gym integration to multi-node training
- **Build custom resources** — Refer to {doc}`/tutorials/creating-resource-server` to create custom resource servers for your specific tasks

## Resources

- {doc}`/about/concepts/training-integration-architecture`
- [NeMo RL Gym tests](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/models/generation/test_vllm_generation.py)
