(training-verification-validate)=

# Validate Verification

Check that your chosen resource server is working correctly before large-scale rollout collection.

:::{card}

**Goal**: Confirm verification produces useful reward signals

^^^

**Quick validation** (5-10 minutes):
1. Collect 20-50 test rollouts
2. Check reward distribution
3. Spot-check high/low examples
4. Verify rewards match your intuition

:::

---

## Quick Validation Checklist

Run a small test and check these indicators:

- [ ] **Distribution is reasonable** - Not all 0.0 or 1.0
- [ ] **High rewards look good** - Best examples are actually good
- [ ] **Low rewards look bad** - Worst examples are actually bad  
- [ ] **Rewards match intuition** - Your assessment aligns with scores

**Time investment**: 5-10 minutes of validation saves hours of bad training data

---

## Step 1: Collect Sample Rollouts

Start with a small test batch:

```bash
# Collect 50 test rollouts
ng_collect_rollouts \
  +input_jsonl_fpath=test_tasks.jsonl \
  +output_jsonl_fpath=test_rollouts.jsonl \
  +num_samples=50
```

**Why 50?**
- Large enough to see distribution
- Small enough to review manually
- Fast enough for quick iteration

---

## Step 2: Check Distribution

View aggregated metrics to see overall reward pattern:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=test_rollouts.jsonl
```

**Example output**:
```
Aggregated results:
  reward: mean=0.64, std=0.23, min=0.0, max=1.0
  accuracy: mean=0.64
```

### What to Look For

:::::{tab-set}

::::{tab-item} Good Patterns ✅

**Varied distribution** (most common):
```
mean=0.64, std=0.23
Distribution: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] all represented
```
✅ **Interpretation**: Verification distinguishes quality well

**High performance** (task is easy):
```
mean=0.89, std=0.15
Distribution: Mostly 0.8-1.0
```
✅ **Interpretation**: Model is good at this task, okay for SFT

**Low performance** (task is hard):
```
mean=0.31, std=0.19
Distribution: Mostly 0.0-0.4
```
✅ **Interpretation**: Task is challenging, may need better model or easier tasks

::::

::::{tab-item} Red Flags ❌

**All rewards identical**:
```
mean=0.0, std=0.0
Distribution: All 0.0
```
❌ **Problem**: Verification always fails or is broken

**All rewards = 1.0**:
```
mean=1.0, std=0.0  
Distribution: All 1.0
```
❌ **Problem**: Verification always succeeds (too lenient) or trivial task

**All rewards clustered**:
```
mean=0.48, std=0.03
Distribution: All 0.45-0.51
```
❌ **Problem**: Verification is noisy, not discriminative

**Binary but continuous expected**:
```
Distribution: Only 0.0 and 1.0, no intermediate values
```
❌ **Problem**: Wrong server for training algorithm (need continuous for DPO/PPO)

::::

:::::

### Interpreting Standard Deviation

**Std dev > 0.20**: ✅ Highly discriminative (good for all training)  
**Std dev 0.10-0.20**: ✅ Moderately discriminative (acceptable)  
**Std dev < 0.10**: ❌ Low discrimination (check verification)

---

## Step 3: Spot-Check Examples

Manually review best and worst examples to validate reward accuracy:

```python
import json

rollouts = [json.loads(line) for line in open('test_rollouts.jsonl')]

# Find highest reward
best = max(rollouts, key=lambda r: r['reward'])
print(f"\n=== HIGHEST REWARD: {best['reward']:.3f} ===")
print("Input:", best['responses_create_params']['input'][-1]['content'][:200])
print("\nOutput:", best['response']['output_text'][:500])

# Find lowest reward
worst = min(rollouts, key=lambda r: r['reward'])
print(f"\n=== LOWEST REWARD: {worst['reward']:.3f} ===")
print("Input:", worst['responses_create_params']['input'][-1]['content'][:200])
print("\nOutput:", worst['response']['output_text'][:500])
```

### Validation Questions

**For high-reward example**:
- ✅ Is this actually a good/correct response?
- ✅ Does it deserve the high score?
- ❌ Are there obvious errors that should lower the score?

**For low-reward example**:
- ✅ Is this actually a bad/incorrect response?
- ✅ Does it deserve the low score?
- ❌ Is it actually correct but scored wrong?

**Red flags**:
- High reward given to obviously wrong answer → Format mismatch or extraction bug
- Low reward given to obviously correct answer → Verification too strict or broken
- Similar quality responses with very different scores → Inconsistent verification

---

## Step 4: Manual Comparison

Pick 5-10 examples and score them yourself, then compare with server:

```python
# Select random samples
import random
samples = random.sample(rollouts, 5)

print("Manual Review:")
for i, rollout in enumerate(samples, 1):
    print(f"\n--- Example {i} ---")
    print("Output:", rollout['response']['output_text'][:200])
    print(f"Server reward: {rollout['reward']:.3f}")
    
    # Your manual assessment
    manual_score = float(input("Your score (0.0-1.0): "))
    diff = abs(manual_score - rollout['reward'])
    
    if diff < 0.2:
        print(f"✅ Aligned (diff={diff:.3f})")
    else:
        print(f"⚠️ Misaligned (diff={diff:.3f})")
```

**Good alignment**: Differences < 0.2

**Investigate if**: Consistent differences > 0.3 (verification may be misconfigured)

---

## Common Issues and Solutions

### Issue 1: All Rewards = 0.0

**Symptoms**: Every rollout gets reward 0.0

**Common causes**:
- Format mismatch (answer extraction failing)
- Verification logic broken
- Task too hard for model
- Configuration error

**Debug steps**:

```python
# Check verification details
rollout = rollouts[0]
print("Expected answer:", rollout.get('verifier_metadata', {}).get('expected_answer'))
print("Agent response:", rollout['response']['output_text'])

# Check for extraction in server logs
# Look for: "Could not extract answer" or similar
```

**Solutions**:
1. **Check format**: Verify agent response format matches what server expects
2. **Check extraction**: Review server's answer extraction logic
3. **Check configuration**: Verify `grading_mode` or extraction settings
4. **Try lenient mode**: If server has one (e.g., mcqa `lenient_boxed`)

---

### Issue 2: All Rewards = 1.0

**Symptoms**: Every rollout gets perfect score

**Common causes**:
- Task too easy for model
- Verification too lenient
- Bug in verification logic

**Debug steps**:

```python
# Check if examples are actually correct
for rollout in rollouts[:5]:
    print("Expected:", rollout.get('verifier_metadata', {}).get('expected_answer'))
    print("Got:", rollout['response']['output_text'][:100])
    print("Reward:", rollout['reward'])
    print("---")
```

**Solutions**:
1. **Make task harder**: Use more challenging dataset
2. **Check verification**: Review verify() logic for bugs
3. **Try stricter mode**: If server has one (e.g., mcqa `strict_single_letter_boxed`)

---

### Issue 3: Rewards Don't Match Quality

**Symptoms**: Good responses get low scores, bad responses get high scores

**Common causes**:
- Answer extraction regex doesn't match agent format
- Expected answer format mismatch
- Judge model bias (for LLM judge servers)

**Debug steps**:

```python
# For servers with extraction
rollout = rollouts[0]

# Check what was extracted
if 'extracted_answer' in rollout:
    print("Extracted:", rollout['extracted_answer'])
    print("Expected:", rollout.get('verifier_metadata', {}).get('expected_answer'))
    print("Full response:", rollout['response']['output_text'])
```

**Solutions**:
1. **Custom regex**: Use `template_metadata.output_regex` for per-task patterns
2. **Change format**: Adjust agent prompt to match expected format
3. **Different server**: Try server with more flexible matching

---

### Issue 4: Distribution Too Narrow

**Symptoms**: All rewards clustered around 0.5, std dev < 0.1

**Common causes**:
- Verification is noisy/random
- Task difficulty matches model capability exactly
- Judge model is uncertain

**Debug steps**:

```bash
# Check if rewards are actually random
python scripts/print_aggregate_results.py +jsonl_fpath=test_rollouts.jsonl

# Look for patterns in distribution
```

**Solutions**:
1. **Change server**: Try different verification approach
   - From LLM judge → deterministic (library_judge_math)
   - From continuous → binary (clearer signal)
2. **Adjust task difficulty**: Make tasks easier or harder
3. **Check judge configuration**: If using LLM judge, verify judge_prompt_template

---

## Server-Specific Validation

### Binary Servers (mcqa, comp_coding, instruction_following)

**Expected pattern**:
```
Rewards: Mostly 0.0 or 1.0
Distribution: Two peaks
```

**Validation focus**:
- ✅ Extraction patterns work for your format
- ✅ High accuracy if task is appropriate for model
- ⚠️ Consider continuous server for DPO/PPO

**Good indicators**:
- Mean between 0.3-0.8 (task is appropriate difficulty)
- Clear correct/incorrect distinction
- No strange intermediate values (0.3, 0.7, etc.)

---

### Continuous Servers (library_judge_math, equivalence_llm_judge)

**Expected pattern**:
```
Rewards: Spread across 0.0-1.0
Distribution: Multiple values represented
```

**Validation focus**:
- ✅ Partial credit makes sense (not all-or-nothing)
- ✅ Quality differences are captured
- ✅ Std dev > 0.15 (discriminative enough)

**Good indicators**:
- Smooth distribution (not just 0.0/1.0)
- High-reward examples clearly better than low-reward
- Intermediate scores for partial correctness

---

### Multi-Metric Servers (multineedle, library_judge_math)

**Expected pattern**:
```
{
  "reward": 0.82,
  "accuracy": 0.82,
  "set_overlap": 0.95
}
```

**Validation focus**:
- ✅ All metrics tracked and aggregated
- ✅ Primary reward (for training) is sensible
- ✅ Additional metrics provide useful insights

**Good indicators**:
- Multiple numeric fields returned
- Primary reward aligns with overall quality
- Additional metrics reveal trade-offs

---

## Switching Servers

If validation reveals issues, consider switching to a different server:

### From Binary to Continuous

**When to switch**:
- Need richer signal for DPO/PPO
- Want to capture partial correctness
- Binary too harsh for your task

**Example**: `mcqa` → `equivalence_llm_judge`

**What changes**:
- Rewards spread across 0.0-1.0 (not just endpoints)
- Can create preference pairs with clear gaps
- More expensive (LLM judge calls)

---

### From Continuous to Binary

**When to switch**:
- Want cleaner SFT data (correct examples only)
- Continuous scores are too noisy
- Don't need nuanced quality assessment

**Example**: `equivalence_llm_judge` → `mcqa`

**What changes**:
- Clear correct/incorrect distinction
- Faster verification (no LLM calls)
- Lose partial credit information

---

### From Single to Multi-Metric

**When to switch**:
- Want to track multiple quality dimensions
- Need to analyze trade-offs
- Single metric doesn't capture all aspects

**Example**: `comp_coding` → `multineedle` (if applicable to your task)

**What changes**:
- Multiple metrics tracked automatically
- Can filter by different criteria
- More complex reward design

---

## Production Readiness Checklist

Before scaling to full collection (millions of rollouts):

- [ ] **Validated with 50+ sample rollouts**
- [ ] **Spot-checked 10+ high/low examples manually**
- [ ] **Distribution looks reasonable** (std dev > 0.15 for continuous)
- [ ] **Rewards align with manual assessment** (diff < 0.2)
- [ ] **Configuration documented** (which server, what settings)
- [ ] **Team reviewed validation results**
- [ ] **Decided on training algorithm** (SFT/DPO/PPO)
- [ ] **Understand what "good" reward means** for your task

**Time investment**: 1-2 hours of validation can save days of bad training data collection

---

## Validation Examples

### Example 1: Binary Server (mcqa)

```bash
# Collect test rollouts
ng_collect_rollouts +resource_server=mcqa +num_samples=50

# Check distribution
python scripts/print_aggregate_results.py +jsonl_fpath=test_rollouts.jsonl
# Output: mean=0.72, std=0.45 (good - 72% accuracy with binary)

# Spot-check
# High reward (1.0): Agent extracted "B", expected "B" ✅
# Low reward (0.0): Agent extracted "D", expected "B" ✅

# Validation: PASS - Ready for SFT collection
```

---

### Example 2: Continuous Server (library_judge_math)

```bash
# Collect test rollouts
ng_collect_rollouts +resource_server=library_judge_math +num_samples=50

# Check distribution
python scripts/print_aggregate_results.py +jsonl_fpath=test_rollouts.jsonl
# Output: mean=0.58, std=0.28 (good - varied quality)

# Spot-check
# High reward (0.92): Correct answer, efficient reasoning ✅
# Medium reward (0.61): Correct approach, minor error ✅
# Low reward (0.18): Wrong method, incorrect answer ✅

# Validation: PASS - Ready for DPO pairing
```

---

### Example 3: Failed Validation

```bash
# Collect test rollouts  
ng_collect_rollouts +resource_server=equivalence_llm_judge +num_samples=50

# Check distribution
python scripts/print_aggregate_results.py +jsonl_fpath=test_rollouts.jsonl
# Output: mean=0.51, std=0.04 (BAD - too narrow)

# Spot-check
# All examples scored 0.48-0.54 regardless of quality ❌

# Validation: FAIL - Judge is uncertain/noisy
# Action: Switch to different server or debug judge configuration
```

---

## Next Steps

**After successful validation**:

:::{button-ref} ../rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start Collecting Rollouts →
:::

**If validation failed**:

:::{button-ref} ../resource-servers/index
:color: secondary
:outline:
:ref-type: doc

Try Different Server →
:::

**For custom verification**:

:::{button-ref} custom-patterns-cookbook
:color: secondary
:outline:
:ref-type: doc

Build Custom Verification →
:::

Or return to {doc}`index` for verification overview.

