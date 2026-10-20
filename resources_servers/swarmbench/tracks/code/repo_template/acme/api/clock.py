def elapsed_ms(end_ms, start_ms):
    """Milliseconds from start_ms to end_ms. May be negative; never raises.

    Replacing legacy span(start, end), which took SECONDS and returned seconds:
    - Argument order: END comes FIRST.
    - Units: milliseconds in and out. Convert the legacy second inputs with * 1000 and the result with // 1000.
      Example: span(a, b) is elapsed_ms(b * 1000, a * 1000) // 1000. Convert exactly the expressions the old code
      passed (e.g. span(x // 1000, y // 1000) becomes elapsed_ms(y // 1000 * 1000, x // 1000 * 1000) // 1000).
    - Errors: legacy raised ValueError when end < start; this returns a negative number. Where the old code caught
      ValueError, test `< 0` and do what the except branch did.
    """
    return end_ms - start_ms


def offset(ts_ms, delta_ms):
    """Timestamp ts_ms (milliseconds) moved by delta_ms milliseconds.

    Replacing legacy shift(ts, minutes=0, hours=0), which took and returned SECONDS (minutes is the 2nd
    positional argument): shift(ts, m, h) is offset(ts * 1000, (m * 60 + h * 3600) * 1000) // 1000.
    """
    return ts_ms + delta_ms
