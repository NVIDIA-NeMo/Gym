# Vendored IFEval scorer

`instructions.py`, `instructions_registry.py`, and `instructions_util.py` are
vendored verbatim from the IHEval repository
(<https://github.com/ytyz1307zzh/IHEval>, `src/rule_following/evaluate/`),
which in turn adapts the **IFEval** instruction-following checkers from Google
Research (<https://github.com/google-research/google-research/tree/master/instruction_following_eval>).

License: Apache-2.0.

They are imported at verify() time by `app.py` via a `sys.path` shim (the
modules use bare ``import instructions`` / ``import instructions_util``), so
they are kept unmodified rather than rewritten as a relative-import package.
