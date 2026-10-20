from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    ok: bool
    value: object = None
    error: str = ""
