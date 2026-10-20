from acme.api._data import SETTINGS


class MissingSetting(KeyError):
    pass


_MISSING = object()


def _parse_bool(text):
    return text.strip().lower() in ("1", "true", "yes", "on")


_KINDS = {"str": str, "int": int, "float": float, "bool": _parse_bool}


def setting(path, default=_MISSING, kind="str"):
    """Read setting "section.key".

    If present, returns the raw string converted by kind ("str", "int", "float", or "bool",
    where bool parses 1/true/yes/on as True). If absent, returns default unchanged, or raises
    MissingSetting when no default was given.

    Replacing legacy get(section, key, default=None, cast=None):
    - Path: one "section.key" string, e.g. get("db", "port") is setting("db.port", ...). For variables, build it
      with section + "." + key.
    - Missing settings: legacy returned default (None if not passed) and never raised. Always pass a default:
      setting(path, None) and then handle None like the old code did.
    - Casting: legacy applied cast(raw) (any callable) only when the setting was present, never to the default.
      Keep that: raw = setting(path, None); value = default if raw is None else cast(raw).
    - Bool trap: legacy cast=bool means Python bool(raw), so any non-empty string, including "false", "no" and "0",
      is True. kind="bool" parses the text and is NOT equivalent; to keep behavior use bool(raw) or `if raw:`.
    """
    if path not in SETTINGS:
        if default is _MISSING:
            raise MissingSetting(path)
        return default
    return _KINDS[kind](SETTINGS[path])
