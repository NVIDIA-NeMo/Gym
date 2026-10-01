# Strategic Bench - NeMo Gym Resource Server

LLMs are largely trained to be cooperative and helpful, with
instruction tuning and reinforcement learning from human feedback (RLHF) optimizing models to provide direct answers,
accommodate user requests, and prioritize immediate helpfulness. However, many real-world settings such as negotiation,
tutoring or therapy require strategic behavior from agents: a negotiator plans which concessions to make across many turns
rather than optimizing each response independently, a tutor may refuse to provide direct answers to encourage learning,
and a therapist withholds preliminary conclusions to gather unbiased information. Instead of strategic behavior in these
tasks, LLMs lose track of goals across multi-turn interactions, contradict themselves, and reveal private information too
early rather than controlling when and what to disclose. Moreover, when placed in out-of-distribution dialogue settings,
LLMs tend to revert to their pretrained, base behaviors, raising reliability concerns during deployment.

This training and evaluation suite includes 0 strategic dialogue tasks inspired by Harvard Program on Negotiation materials to train models on multi-turn, long-horizon dialogue tasks evaluated by LLM-as-a-Judge.

## Features
- Computes final reward and metrics upon conversation termination.

## Setup
Use the NeMo Gym framework utilities to mount `app.py` as a simulated REST endpoint for tool execution.

## Testing
Run the provided tests:
```bash
pytest tests/
```