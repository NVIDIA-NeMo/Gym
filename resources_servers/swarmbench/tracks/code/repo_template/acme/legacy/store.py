# DEPRECATED: being removed. See the "Replacing legacy" docstrings in acme/api/store.py and MIGRATION.md.
from acme.api import store as _store

_MISSING = object()


def fetch(key, default=_MISSING):
    """Return (value, version) for key. If missing: return default if given, else raise KeyError."""
    rec = _store.lookup(key)
    if rec is None:
        if default is _MISSING:
            raise KeyError(key)
        return default
    return (rec.value, rec.version)


def put(key, value):
    """Store value and return the new version number (int)."""
    return _store.save(key, value).version
