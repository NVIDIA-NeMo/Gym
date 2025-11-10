(training-integration-nemo-rl-sft)=

# SFT Training Guide

Train with **Supervised Fine-Tuning** using high-quality demonstration data.

**Best for**: High-reward rollouts, imitation learning

---

:::{note}
This guide is under development. For now, refer to the {doc}`index` for an overview of NeMo RL integration patterns and {ref}`training-rollout-sampling-sft` for data preparation guidance.
:::

## Overview

SFT (Supervised Fine-Tuning) trains models on high-quality demonstrations:

- Uses rollouts with high rewards (typically ≥ 0.9)
- Standard supervised learning approach
- Best for imitation of successful behaviors
- Requires high-quality training data

## Quick Start

Coming soon.

## Related Resources

- {doc}`index` - NeMo RL Integration Overview
- {ref}`training-rollout-sampling-sft` - SFT Data Preparation
- {doc}`advanced` - Advanced Integration Patterns
- {ref}`training-rollout-collection` - Rollout Collection Guide
