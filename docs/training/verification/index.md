(training-verification)=

# Verification

Validate that your resource server's verification logic produces useful reward signals, or build custom verification for specialized domains.

:::{seealso}
**Already chose a resource server?** You're in the right place—this section helps you validate it works and customize if needed.

**Haven't chosen yet?** Start with {doc}`../resource-servers/index` to select a resource server first.
:::

---

## Choose Your Path

:::::{grid} 1 1 2 2
:gutter: 3

::::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Validate Verification
**Most users** - Check your resource server works

After collecting sample rollouts, validate that reward signals are useful for training.

+++
{bdg-primary}`Start here`

:::{button-ref} validate-verification
:color: primary
:outline:
:ref-type: doc

Validation Guide →
:::
::::

::::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Verification
**Advanced** - For specialized domains

Create custom verification logic with copy-paste patterns and multi-objective techniques.

+++
{bdg-secondary}`Advanced`

:::{button-ref} custom-patterns-cookbook
:color: secondary
:outline:
:ref-type: doc

Custom Patterns →
:::
::::

:::::

---

## When to Validate

Validate verification after collecting your first sample rollouts:

```
1. Choose resource server → See resource-servers/
   ↓
2. Collect 50 sample rollouts → See rollout-collection/
   ↓
3. Validate verification → You are here
   ↓
4. Scale to full collection → See rollout-collection/
   ↓
5. Prepare training data → See datasets/
```

**Why validate?** Confirms reward signals are:
- Discriminative (different quality → different scores)
- Aligned (high scores = actually good responses)
- Appropriate (distribution matches training algorithm needs)

**Time**: 5-10 minutes with validation checklist

---

## Validation Workflow

:::{button-ref} validate-verification
:color: primary
:outline:
:ref-type: doc

Full Validation Guide →
:::

**Quick checklist**:
1. Collect 50 test rollouts
2. Check reward distribution (not all 0.0 or 1.0)
3. Spot-check high/low examples
4. Confirm alignment with your quality assessment

**Common validation tasks**:
- Verify extraction patterns work for your format
- Check that reward ranges match training algorithm (binary for SFT, continuous for DPO/PPO)
- Identify and debug issues (all zeros, misaligned scores)
- Switch servers if needed

---

## Custom Verification

For specialized domains not covered by the 13 built-in resource servers:

:::{button-ref} custom-patterns-cookbook
:color: secondary
:outline:
:ref-type: doc

Custom Patterns Cookbook →
:::

**Six copy-paste patterns**:
1. Exact Match (MCQA-style)
2. LLM Judge (semantic equivalence)
3. Code Execution (test-based)
4. Schema Validation (structured outputs)
5. Multi-Objective (multiple metrics)
6. Hybrid (fallback strategies)

**Advanced**: {doc}`multi-objective-scoring` for combining multiple reward signals

---

## Next Steps

**Most users** (validating existing server):

:::{button-ref} validate-verification
:color: primary
:outline:
:ref-type: doc

Validate Verification →
:::

**Advanced users** (building custom):

:::{button-ref} custom-patterns-cookbook
:color: secondary
:outline:
:ref-type: doc

Custom Patterns Cookbook →
:::

```{toctree}
:hidden:
:maxdepth: 1

validate-verification
custom-patterns-cookbook
multi-objective-scoring
```
