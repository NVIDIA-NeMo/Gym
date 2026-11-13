# FM01 Codex version-output fixture

`fm01-codex-version-session.jsonl` contains the final command and polling results
from FM01 run `0d77cf2a8ca94a4d8e03f44ce72705d2`, saved in
`results/harbor/gym-fm-01/rollouts.jsonl` under
`ng_agent_observations.records[0].conversation`.

The four saved conversation items are wrapped as session `response_item` records
for reader tests. Their call IDs, arguments and output are unchanged. The raw
session file was not retained by that run; these are reconstructed record wrappers,
not a copy of the original JSONL file. Timestamps and unrelated conversation items
are omitted. The fixture exercises output/completion reading, not installation.

# FM02 oracle trajectory fixture

`fm02-oracle-trajectory.json` is a byte-for-byte copy of the ATIF trajectory from
FM02 supplied-solution run `fab9aa981d30476abcd33b886433e051`, saved at
`results/harbor/validation/gym-fm-02/verifier/session-4d7dc2338f154f8f9d18cc3dd06bd37f/trajectory.json`.
It contains the actual reference solution's output tail, not a coding-agent tool
trace. The reader previously removed that message and treated the remaining
metadata as a reason to skip the oracle-log fallback.

The original full `/logs/agent/oracle.txt` was not retained on the host. The
regression test reconstructs a temporary oracle log using the output tail from
this trajectory. This checks evidence discovery and delivery, not a new Gym run
or a judge verdict. No synthetic tool calls are added to the saved trajectory.
