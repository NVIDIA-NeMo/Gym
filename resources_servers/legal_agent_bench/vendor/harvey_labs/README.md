# Harvey LAB attribution

This directory contains a minimal runtime adaptation of
[Harvey LAB](https://github.com/harveyai/harvey-labs), pinned to commit
`f46ef86e4788545622db25dcffa3aebb7a139929` and distributed under Harvey's
MIT license in `LICENSE`.

NeMo Gym modifications add an asynchronous protocol adapter for the local Gym
model server selected by the Harbor agent,
path-constrained file tools, pagination and telemetry, and a Harbor-oriented
system prompt. `lab_harbor/scoring.py` is an attributed adaptation of upstream
`evaluation/scoring.py`; its module docstring lists each intentional behavioral
difference. `lab_harbor/judge.py` contains the Gym-specific judge transport.
The agent can also use `bash` within the Harbor task container.
The benchmark tasks and the public `docx`, `pptx`, and `xlsx` skills are not
checked into this repository; `prepare.py` downloads and verifies that pinned
source revision.
