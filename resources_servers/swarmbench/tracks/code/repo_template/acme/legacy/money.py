# DEPRECATED: being removed. See the "Replacing legacy" docstrings in acme/api/money.py and MIGRATION.md.
from acme.api._data import RATES


class ConversionError(Exception):
    pass


def convert(amount, dst, src="USD"):
    """Convert amount in major units (e.g. 12.5 dollars) from src to dst; returns major units.

    The amount is first rounded to whole cents with round(amount * 100); the converted cents
    are floored. Raises ConversionError for an unknown currency.
    """
    if src not in RATES or dst not in RATES:
        raise ConversionError(f"unknown currency: {src}->{dst}")
    cents = round(amount * 100)
    return cents * RATES[dst] // RATES[src] / 100
