from dataclasses import dataclass

from acme.api import _data


@dataclass(frozen=True)
class Record:
    key: str
    value: object
    version: int


_rows = {}


def reset():
    _rows.clear()
    for key, (value, version) in _data.STORE.items():
        _rows[key] = Record(key, value, version)


def lookup(key):
    """Return the Record for key, or None if the key does not exist. Never raises.

    Replacing legacy fetch(key, default):
    - Legacy returned a (value, version) tuple. A Record cannot be unpacked or indexed: use rec.value
      (legacy row[0]) and rec.version (legacy row[1]).
    - A missing key: legacy raised KeyError (no default) or returned the default exactly as given
      (e.g. None, or a tuple like ("guest", 0)). lookup returns None, so write `if rec is None:` and reproduce
      what the old code did (the except KeyError branch, or the values from the default tuple).
    """
    return _rows.get(key)


def save(key, value):
    """Store value under key and return the new Record (version = previous version + 1, or 1 if new).

    Replacing legacy put(key, value), which returned the new version int: use save(key, value).version.
    """
    old = _rows.get(key)
    rec = Record(key, value, old.version + 1 if old else 1)
    _rows[key] = rec
    return rec


reset()
