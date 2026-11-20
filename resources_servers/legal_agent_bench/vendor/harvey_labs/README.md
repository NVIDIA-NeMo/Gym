# Harvey LAB attribution

The adjacent `../../harness/` and `../../lab_harbor/` directories contain a
minimal runtime adaptation of
[Harvey LAB](https://github.com/harveyai/harvey-labs), pinned to commit
`f46ef86e4788545622db25dcffa3aebb7a139929` and distributed under Harvey's
MIT license in `LICENSE`.

NeMo Gym modifications add an asynchronous protocol adapter for the local Gym
model server selected by the Harbor agent,
path-constrained file tools, pagination and telemetry, and a Harbor-oriented
system prompt. `../../lab_harbor/scoring.py` is an attributed adaptation of upstream
`evaluation/scoring.py`; its module docstring lists each intentional behavioral
difference. `../../lab_harbor/judge.py` contains the Gym-specific judge transport.
The agent can also use `bash` within the Harbor task container.
The benchmark tasks and the public `docx`, `pptx`, and `xlsx` skills are not
checked into this repository; `prepare.py` downloads and verifies that pinned
source revision.

## Licenses and source provenance

This directory keeps license and attribution files only. Original Harvey AI
portions remain MIT-licensed under the unchanged `LICENSE`. NVIDIA-authored
code and NVIDIA modifications are Apache-2.0 licensed; modified source files
retain the original notices and add a separate NVIDIA modifications block.
The upstream software is not relicensed. The component is also listed in the
repository's [ATTRIBUTIONS.md](../../../../ATTRIBUTIONS.md).

The following upstream files at the pinned revision define the adapted code
and protocols. Paths on the right are relative to `legal_agent_bench/`.

| Upstream source | Local adaptation |
| --- | --- |
| [`harness/adapters/base.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/adapters/base.py) | `harness/adapters/base.py`; protocol used by `harness/adapters/openai_compatible.py` |
| [`harness/adapters/openai.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/adapters/openai.py) | `harness/adapters/openai_compatible.py` |
| [`harness/agent_loop.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/agent_loop.py) | `legal_harbor_agent.py` |
| [`harness/tools.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/tools.py) | `lab_harbor/tools.py`, `lab_harbor/container_tool_runner.py` |
| [`harness/system_prompt.md`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/system_prompt.md) | `harness/system-prompt.md` |
| [`evaluation/scoring.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/scoring.py) | `lab_harbor/scoring.py`; scoring contract used by `verifier.py` |
| [`evaluation/prompts/rubric_criterion.txt`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/prompts/rubric_criterion.txt) | rubric prompt in `lab_harbor/judge.py` |

The system-prompt content is preserved during the package relocation. Its
MIT attribution is recorded here without inserting notice text into the
agent-visible prompt.
