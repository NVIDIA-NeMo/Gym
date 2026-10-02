# FACTS Search agent

This agent is the public FACTS Search-On loop, not a general browsing harness. A search-enabled hop is one model turn containing one or more parallel `brave_search` calls. It permits seven such hops. If the seventh hop still contains tool calls, it executes all of them and then makes the published final tool-free request: `Please provide a final answer based on the information gathered so far.`

The full model/tool trajectory, hop count, query count, and whether the forced-final path ran are retained in the Gym rollout.

All parallel search calls in a hop execute before the next policy turn. Invalid tool arguments and non-2xx search replies are preserved as failed tool-call records rather than terminating the episode. An ordinary assistant answer ends the loop immediately; incomplete or empty output is retained without fabricating a final answer.
