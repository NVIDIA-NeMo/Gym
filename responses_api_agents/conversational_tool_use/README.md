# Conversational Tool Use

The conversational tool-use workflow is implemented by four Responses API agents:

| Stage | Implementation | Config |
|---|---|---|
| Conversation simulation | [`simulation`](simulation) | [`simulation.yaml`](simulation/configs/simulation.yaml) |
| Domain generation | [`domain_generation`](domain_generation) | [`domain_generation.yaml`](domain_generation/configs/domain_generation.yaml) |
| Policy and tool generation | [`policy_tool_generation`](policy_tool_generation) | [`policy_tool_generation.yaml`](policy_tool_generation/configs/policy_tool_generation.yaml) |
| Scenario generation | [`scenario_generation`](scenario_generation) | [`scenario_generation.yaml`](scenario_generation/configs/scenario_generation.yaml) |

The simulation resource server and shared workflow documentation live in
[`resources_servers/conversational_tool_use_simulation`](../../resources_servers/conversational_tool_use_simulation).
