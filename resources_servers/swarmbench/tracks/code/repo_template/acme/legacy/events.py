# DEPRECATED: being removed. See the "Replacing legacy" docstrings in acme/api/events.py and MIGRATION.md.
from acme.api.events import HANDLERS


def emit(topic, payload):
    """Call every handler of topic synchronously; return their replies in registration order."""
    return [handler(payload) for handler in HANDLERS.get(topic, [])]
