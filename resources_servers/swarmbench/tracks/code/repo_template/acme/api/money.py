from acme.api._data import RATES
from acme.api.result import Result


def exchange(src, dst, cents):
    """Convert an integer number of cents from currency src to currency dst.

    Returns Result(ok=True, value=<int cents in dst, floored>), or Result(ok=False, error=...) for an unknown
    currency or non-int cents. Never raises.

    Replacing legacy convert(amount, dst, src="USD"), which took and returned major units (dollars):
    - Argument order: src comes FIRST, then dst, then the amount. Legacy defaulted src to "USD", so a call like
      convert(x, "EUR") becomes exchange("USD", "EUR", ...); convert(x, "EUR", "GBP") becomes exchange("GBP", "EUR", ...).
    - Units: pass round(amount * 100) (an int, rounded exactly like legacy did). res.value is int cents, so
      res.value / 100 is the float that legacy returned.
    - Errors: legacy raised ConversionError; this returns ok=False instead. Where the old code caught
      ConversionError, check `if not res.ok:` and do what the except branch did.
    """
    if src not in RATES or dst not in RATES:
        return Result(False, None, f"unknown currency: {src}->{dst}")
    if not isinstance(cents, int) or isinstance(cents, bool):
        return Result(False, None, "cents must be an int")
    return Result(True, cents * RATES[dst] // RATES[src])
