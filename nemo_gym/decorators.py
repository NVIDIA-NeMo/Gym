import functools

import rich


def experimental(fn):
    """Decorator that prints an experimental warning before the function runs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rich.print(
            f"[yellow]Warning:[/yellow] [bold]{fn.__name__}[/bold] is experimental and may change or be removed without notice."
        )
        return fn(*args, **kwargs)

    return wrapper
