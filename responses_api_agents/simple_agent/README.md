# Description

Simple Agent supports an explicit outcome policy for confirmed policy-model response timeouts on its legacy `/run` route. The default is `retry`: an ungraded, nonterminal failure eligible for Collection’s existing bounded retry on resume. It does not start another immediate model call.

```yaml
outcome_policy:
  by_failure_kind:
    model_policy_response_timeout: count_as_zero
```

`fail` stops the rollout ungraded; `count_as_zero` stops it with an explicit zero. All three actions retain the same classified failure, IDs and causes. Classification requires every observed attempt of the final model operation to have timed out waiting for a response after a confirmed completed upload. Missing phase or attempt evidence, failed uploads, connection/pool timeouts and mixed failures do not qualify. Older reports without upload evidence therefore cannot activate this scoring rule. Model-request and episode budgets remain separate.

Status rules take precedence over kind rules, then `default_action`. The adapter selects the event’s own status; it does not search inner causes. `verify` is rejected when loading this boundary’s policy. Environment Server policies retain their existing verification behavior.

`final_on_model_timeout: true` remains a compatibility alias for the rule above; explicit false selects `retry`. The alias is resolved once into `outcome_policy` and excluded from the serialized resolved configuration. An explicitly conflicting timeout rule or default is rejected. Prefer `outcome_policy` in new configurations.


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
