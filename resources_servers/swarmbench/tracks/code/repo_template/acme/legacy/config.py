# DEPRECATED: being removed. See the "Replacing legacy" docstrings in acme/api/config.py and MIGRATION.md.
from acme.api._data import SETTINGS


def get(section, key, default=None, cast=None):
    """Raw string for "section.key", passed through cast(raw) if cast is given.

    Returns default (not cast) when the setting is absent.
    """
    raw = SETTINGS.get(f"{section}.{key}")
    if raw is None:
        return default
    return cast(raw) if cast else raw
