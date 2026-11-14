# Harvey LAB attribution

The adjacent `../../harness/` and `../../lab_harbor/` directories contain a
minimal runtime adaptation of
[Harvey LAB](https://github.com/harveyai/harvey-labs), pinned to commit
`9e9fa7fe126103de1fceb2227b7c70bf59b67f73` and distributed under Harvey's
MIT license in `LICENSE`. The upstream Pandoc installation helper is retained
at `../../install-pandoc.sh`.

NeMo Gym adaptations provide path-constrained file tools, pagination, telemetry,
and a task system prompt. The session agent calls Gym's model server directly.
`../../lab_harbor/scoring.py` is an attributed adaptation of upstream
`lab_core/evaluation/scoring.py`; its module docstring lists each intentional behavioral
difference. `../../lab_harbor/judge.py` contains the Gym-specific judge transport.
The agent can also use `bash` within the Harbor task container.
The benchmark tasks and the public `docx`, `pptx`, and `xlsx` skills are not
checked into this repository; `prepare.py` downloads and verifies that pinned
source revision.

The runtime refresh follows the reasoning-first judge schema, DOCX margin-comment
extraction, explicit finish tool, and official averaged dual-judge task scoring.
Gym retains its configured HTTP transport, per-criterion tracing, deterministic
deliverable matching, and isolated verifier. These adaptations are listed in
module docstrings; no provider SDK orchestration is imported.

## Licenses and source provenance

This directory keeps license and attribution files only. Original Harvey AI
portions remain MIT-licensed under the unchanged `LICENSE`. NVIDIA-authored
code and NVIDIA modifications are Apache-2.0 licensed; modified source files
retain the original notices and add a separate NVIDIA modifications block.
The upstream software is not relicensed. The component is also listed in the
repository's [ATTRIBUTIONS.md](../../../../ATTRIBUTIONS.md).

The following original upstream files define the adapted code and protocols.
Later refreshes record their own exact source revisions in file headers. Paths on the right are relative to `legal_agent_bench/`.

| Upstream source | Local adaptation |
| --- | --- |
| [`harness/tools.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/tools.py) | `lab_harbor/tools.py`, `lab_harbor/container_tool_runner.py` |
| [`harness/system_prompt.md`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/harness/system_prompt.md) | `harness/system-prompt.md` |
| [`evaluation/scoring.py`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/scoring.py) | `lab_harbor/scoring.py`; scoring contract used by `verifier.py` |
| [`evaluation/prompts/rubric_criterion.txt`](https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/prompts/rubric_criterion.txt) | rubric prompt in `lab_harbor/judge.py` |

The system-prompt content is preserved during the package relocation. Its
MIT attribution is recorded here without inserting notice text into the
agent-visible prompt.

The runtime refresh also adapts `lab_core/harness/tools.py`,
`lab_core/harness/system_prompt.md`, `lab_core/evaluation/scoring.py`,
`lab_core/evaluation/prompts/rubric_criterion.txt`,
`lab_core/sandbox/parsers/parse_doc.py`, and the Pandoc installer from
[revision 9e9fa7fe126103de1fceb2227b7c70bf59b67f73](https://github.com/harveyai/harvey-labs/tree/9e9fa7fe126103de1fceb2227b7c70bf59b67f73).
The affected local files are `lab_harbor/tools.py`, `harness/system-prompt.md`,
`lab_harbor/scoring.py`, `lab_harbor/judge.py`,
`lab_harbor/container_tool_runner.py`, and `install-pandoc.sh`. Their source
headers identify each original file and retained implementation.
