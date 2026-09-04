# Constraint verifier

Grades one instruction-following constraint over one trajectory: a list of gradable steps, each with a binary reward,
empty when the constraint's trigger never fired (not applicable). This package is the only implementation; the design
rationale, the semantic rulings and the validation record live in the design recipe
(`agentic-if/recipes/if-constraint-design/verifier/VERIFIER_SPEC.md`, `VALIDATION.md`), which imports this package.

| file | holds | registry |
|---|---|---|
| `core.py` | `Turn`, `ToolCall`, `GradedStep`, `DEFAULT_RESOLVER`, the no-answer policies, text and tool-call helpers | none |
| `matchers.py` | what an obligation checks | `MATCHERS[name] = Matcher(name, check, silent_turn, doc, value_key, witness, violation, examples, instruction_kind)` |
| `triggers.py` | which turns a constraint applies to, keyed by the trigger-dict key: `position`, `tool`, `prev_tool`, `prev_message`, `all_of` | `TRIGGERS[key] = Trigger(key, select, doc, owns, missing, examples)` |
| `templates.py` | which surface is graded: `turn_output`, `reply_output`, `tool_args`, `tool_choice` | `TEMPLATES[name] = Template(name, grade, doc, applies_policy)` |
| `__init__.py` | `grade`, `grade_ext`: template → trigger selection → matcher, with the no-answer policy | — |

## Extending it

- **A matcher** is one `Matcher(...)` entry in `matchers.py`: `check(value, stripped_text) -> (ok, detail)`; the mandatory
  `silent_turn` policy (`SILENT_TURN_FAILS` for a shape that needs an answer, `SILENT_TURN_NOT_GRADABLE` for a rule that
  silence cannot violate, or a function of the value); one doc line; `examples` (values). Optional: `value_key`, `witness`,
  `violation`, `instruction_kind`.
- **A trigger (conditioner)** is one `Trigger(...)` entry in `triggers.py`: `select(turns, trigger, resolver) -> [(turn,
  detail_prefix)]`, the modifier keys it `owns`, one doc line, optional `missing`, and `examples` = `(trigger_dict,
  expected_turn_indices)` on `example_trace()`. `{"all_of": [t1, t2]}` composes triggers; no template branch is needed.
- **A template** is one `Template(...)` entry in `templates.py`.

`../tests/test_verifier_registry.py` is generated from the registries (an entry without examples fails it);
`../tests/test_verifier.py` holds the behavioural tests. Stdlib only.
