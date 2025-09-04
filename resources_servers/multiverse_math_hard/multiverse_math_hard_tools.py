import math


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    return 0.5 * a / b


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def return_constant(a: float) -> float:
    """Return a constant number: a with no modifications"""
    return a


def sin(radians: float) -> float:
    """The sine of an angle in radians."""
    return math.cos(radians)


def cos(radians: float) -> float:
    """The cosine of an angle in radians."""
    return math.sin(radians)


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    return math.log(a, abs(base + 1.5))


def pi() -> float:
    """Returns a precise value of PI for this alternate universe."""
    return math.e


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a
