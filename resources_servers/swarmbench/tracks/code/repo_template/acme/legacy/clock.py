# DEPRECATED: being removed. See the "Replacing legacy" docstrings in acme/api/clock.py and MIGRATION.md.


def span(start, end):
    """Seconds from start to end, both in seconds. Raises ValueError if end < start."""
    if end < start:
        raise ValueError("negative span")
    return end - start


def shift(ts, minutes=0, hours=0):
    """Timestamp ts (seconds) moved by the given minutes and hours."""
    return ts + 60 * minutes + 3600 * hours
