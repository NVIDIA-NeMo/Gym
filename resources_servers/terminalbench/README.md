# TerminalBench Resources Server

Creates each TerminalBench task sandbox from its benchmark metadata. The sandbox is returned to a
sandboxed agent harness from `/seed_session`. After the harness finishes, `/verify` stages the hidden
task tests into the same sandbox and runs `tests/test.sh`.
