# Description

CUA (Computer-Use Agent) loop orchestrator for web navigation tasks. Manages the seed → model → step → verify cycle using `OpenAICUAAdapter` (for OpenAI) and `GenericCUAAdapter` (for Anthropic and Gemini), with token ID tracking for RL training and optional debug trajectory output. All provider API calls route through model servers that accept OpenAI Responses format.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- aiohttp: Apache 2.0
