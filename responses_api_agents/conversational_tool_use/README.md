# Conversational Tool Use

The conversational tool-use workflow is implemented by four Responses API agents:

| Stage | Implementation | Config |
|---|---|---|
| Domain generation | [`domain_generation`](domain_generation) | [`conversational_tool_use_domain_generation.yaml`](domain_generation/configs/conversational_tool_use_domain_generation.yaml) |
| Policy and tool generation | [`policy_tool_generation`](policy_tool_generation) | [`conversational_tool_use_policy_tool_generation.yaml`](policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml) |
| Scenario generation | [`scenario_generation`](scenario_generation) | [`conversational_tool_use_scenario_generation.yaml`](scenario_generation/configs/conversational_tool_use_scenario_generation.yaml) |
| Conversation simulation | [`simulation`](simulation) | [`conversational_tool_use_simulation.yaml`](simulation/configs/conversational_tool_use_simulation.yaml) |

The generation stages run in table order. The complete conversation stack, shared workflow documentation, and runnable
resource-server configs live in
[`resources_servers/conversational_tool_use_simulation`](../../resources_servers/conversational_tool_use_simulation).
