# AA-Briefcase-Lite resources server

The server supplies the standard stateless session endpoint required by NeMo
Gym and grades artifacts independently from policy rollout. Grader-only checks,
prompts, source graphs, and reference artifacts are loaded server-side and do
not enter the agent request.

Binary mode implements all 55 released A/C checks with AA's released system and
user judge prompts. Each check is judged independently and malformed judge
output fails closed. Pairwise mode evaluates the eight released AQ/P criteria
against configured public reference submissions with GDPval's artifact
handling, compatible judge routing, and position-debiased trials. The default
reference is the public `gpt-5-5` example.

AA did not publish its production pairwise prompt, private comparison graph,
or full judge/aggregation details. Pairwise mode is therefore a local
diagnostic. `all` mode reports binary and pairwise metrics separately and also
returns a non-official convenience average. No output from this server is
AA-Briefcase leaderboard comparable.
