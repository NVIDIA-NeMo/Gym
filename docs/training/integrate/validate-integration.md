(integrate-validate-integration)=

# Validate Your Integration

Verify your training framework integration works correctly end-to-end.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
15 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed previous guides in this section
- Working rollout collection and processing

:::

::::

---

## Validation Checklist

Run through this checklist to verify your integration is correct:

- [ ] HTTP endpoint responds correctly
- [ ] Token round-trips are stable
- [ ] Rollouts complete with rewards
- [ ] Token sequences are contiguous
- [ ] Training gradients flow correctly

---

## Test HTTP Endpoint

Verify your OpenAI-compatible endpoint handles all required operations:

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
        
        # Test 3: Tokenization (if available)
        async with session.post(
            f"{base_url}/tokenize",
            json={"prompt": "Hello world"}
        ) as resp:
            if resp.status == 200:
                print("✅ Tokenization endpoint works")
            else:
                print("⚠️  Tokenization endpoint not available (optional)")


# Run the test
asyncio.run(test_endpoint("http://localhost:8000"))
```

**✅ Expected output**:

```text
✅ Model listing works
✅ Chat completions work
✅ Tokenization endpoint works
```

---

## Test Token Round-Trip Stability

Verify that tokenization is stable across encode/decode cycles:

```python
def test_token_roundtrip(tokenizer, test_strings: list[str]):
    """Test that tokenization round-trips are stable."""
    failures = []
    
    for text in test_strings:
        # Encode
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        # Re-encode
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
        for f in failures[:3]:  # Show first 3
            print(f"   '{f['original']}' -> {f['token_ids']} -> '{f['decoded']}' -> {f['re_encoded']}")
        return False
    
    print(f"✅ All {len(test_strings)} round-trips stable")
    return True


# Test with various strings
test_strings = [
    "Hello, world!",
    "The answer is 42.",
    "```python\nprint('hello')\n```",
    "Multi\nline\ntext",
    "Special chars: @#$%",
]
test_token_roundtrip(tokenizer, test_strings)
```

---

## Test End-to-End Rollout

Collect a rollout and verify all fields are present:

```python
async def test_rollout_collection(gym: GymIntegration):
    """Test complete rollout collection."""
    
    # Simple test example
    example = {
        "responses_create_params": {
            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
            "model": "policy",
        },
        "agent_ref": "your_agent_name",
    }
    
    # Collect rollout
    async for rollout in gym.collect_rollouts([example]):
        # Verify structure
        assert "response" in rollout, "Missing response"
        assert "output" in rollout["response"], "Missing output"
        assert "reward" in rollout, "Missing reward"
        
        # Verify reward is valid
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

---

## Test Token Continuity

Verify token sequences align correctly:

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

---

## Test Training Gradient Flow

Verify that processed rollouts produce valid gradients:

```python
import torch


def test_gradient_flow(
    processed_rollout: dict,
    model: torch.nn.Module,
):
    """Test that gradients flow correctly through processed data."""
    
    # Concatenate all trainable token IDs
    trainable_ids = []
    for msg in processed_rollout["message_log"]:
        if msg.get("trainable", False):
            trainable_ids.extend(msg["token_ids"].tolist())
    
    if not trainable_ids:
        print("⚠️  No trainable tokens in rollout")
        return False
    
    # Create a simple forward pass
    input_ids = torch.tensor([trainable_ids])
    
    # Forward pass (adjust for your model interface)
    model.train()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
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

---

## Run Full Validation Suite

Combine all tests into a validation script:

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
        
        # Test 5: Gradient Flow (if model provided)
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

---

## Expected Results

A successful validation produces:

```text
==================================================
Training Framework Integration Validation
==================================================

📡 Testing HTTP endpoint...
✅ Model listing works
✅ Chat completions work
✅ Tokenization endpoint works

🔄 Testing token round-trips...
✅ All 3 round-trips stable

📦 Testing rollout collection...
✅ Rollout collected successfully
   Reward: 1.0
   Trainable outputs: 2

🔗 Testing token continuity...
✅ Token continuity validated across 2 turns

📈 Testing gradient flow...
✅ Gradients flow correctly (loss=2.3456)

==================================================
VALIDATION SUMMARY
==================================================
  http_endpoint: ✅ PASS
  token_roundtrip: ✅ PASS
  rollout_collection: ✅ PASS
  token_continuity: ✅ PASS
  gradient_flow: ✅ PASS

🎉 All validations passed! Your integration is ready.
```

---

## Troubleshooting

### HTTP endpoint tests fail

Refer to {doc}`expose-openai-endpoint` and verify your vLLM configuration.

### Token round-trips unstable

Some tokenizers behave inconsistently. Consider:
- Using `add_special_tokens=False` consistently
- Stripping whitespace before comparison
- Filing an issue with the tokenizer maintainers

### Rollout collection fails

Refer to {doc}`connect-gym-to-training` and check:
- Gym servers are running
- Agent configuration is correct
- Network connectivity between components

### Token continuity fails

Refer to {doc}`process-multi-turn-rollouts` for debugging steps.

---

## Integration Complete

Congratulations! 🎉 Your training framework integration is validated and ready for production use.

**Completed**:

1. ✅ {doc}`expose-openai-endpoint` — HTTP endpoint configured
2. ✅ {doc}`connect-gym-to-training` — Gym integrated into training loop
3. ✅ {doc}`process-multi-turn-rollouts` — Rollouts processed correctly
4. ✅ {doc}`validate-integration` — Integration validated

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Scale Up Training
:link: /tutorials/integrate-training-frameworks/train-with-nemo-rl
:link-type: doc

See how NeMo RL scales Gym integration to multi-node training.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Resources
:link: /tutorials/creating-resource-server
:link-type: doc

Create custom resource servers for your specific tasks.
:::

::::

---

## Reference

- {doc}`/about/concepts/training-integration-architecture`
- [NeMo RL Gym tests](https://github.com/NVIDIA-NeMo/RL/blob/main/tests/unit/models/generation/test_vllm_generation.py)

