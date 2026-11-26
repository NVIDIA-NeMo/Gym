# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run STEM's configured Math-Verify metric in a bounded worker process."""

import asyncio
import ctypes
import logging
import os
import re
import signal
import socket
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig
from sympy import Expr, FiniteSet, Rational, Tuple

from nemo_gym.telemetry import MeteredSemaphore, resources_operation


if TYPE_CHECKING:
    from resources_servers.stem_judge.app import MathVerifyConfig


# Math-Verify extracts expressions from prose and can silently drop notation even
# with string fallback disabled. Admit whole scalar expressions before calling it.
_DECIMAL = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_DENOMINATOR = r"[1-9][0-9]*"
_FRACTION = rf"(?:[0-9]+/{_DENOMINATOR}|\\frac\{{-?[0-9]+\}}\{{{_DENOMINATOR}\}})"
# e and i have conventional constant/exponent interpretations.
_VARIABLE = r"[a-df-hj-z]"
_NUMBER = rf"(?:{_FRACTION}|{_DECIMAL})"
_TERM = rf"(?:{_FRACTION}|{_DECIMAL}{_VARIABLE}?|{_VARIABLE})"
_NUMERIC = re.compile(rf"[+-]?{_NUMBER}(?: *[+-] *{_NUMBER})*")
_SCALAR = re.compile(rf"[+-]?{_TERM}(?: *[+-] *{_TERM})*")
_DECIMAL_WITH_POINT = re.compile(r"[0-9]*\.[0-9]+")


def _exact_decimal(match: re.Match[str]) -> str:
    value = Fraction(match[0])
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _parse_scalar(text: str, latex: LatexExtractionConfig, *, numeric: bool = False) -> Expr | None:
    text = text.strip()
    if not (_NUMERIC if numeric else _SCALAR).fullmatch(text):
        return None
    # Parsing decimals as floats loses low-order digits before a sum or difference is evaluated.
    text = _DECIMAL_WITH_POINT.sub(_exact_decimal, text)
    values = parse(r"\boxed{" + text + "}", (latex,), fallback_mode="no_fallback", parsing_timeout=0)
    if len(values) == 1 and isinstance(values[0], Expr) and values[0].is_finite is not False:
        return values[0].doit()
    return None


def _parse_answer(text: str, latex: LatexExtractionConfig) -> tuple[str, Expr | FiniteSet | Tuple] | None:
    text = text.strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    elif len(text) > 1 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    # Parse members separately so the library cannot discard one or confuse a pair with its endpoint set.
    is_set = text.startswith(r"\{") and text.endswith(r"\}")
    is_pair = text.startswith("(") and text.endswith(")")
    if is_set or is_pair:
        members = (text[2:-2] if is_set else text[1:-1]).split(",")
        if is_pair and len(members) != 2:
            return None
        values = [_parse_scalar(member, latex, numeric=True) for member in members]
        if None in values:
            return None
        return "", FiniteSet(*values) if is_set else Tuple(*values)

    if "=" in text:
        name, *terms = text.split("=")
        name = name.strip()
        if not re.fullmatch(_VARIABLE, name):
            return None
        # Only a single variable assigned a numeric value is interchangeable with that value.
        values = [_parse_scalar(term, latex, numeric=True) for term in terms]
        if None in values:
            return None
        if any(values[0] != value for value in values[1:]):
            return None
        return name, values[0]

    value = _parse_scalar(text, latex)
    return ("", value) if value is not None else None


def _matches(reference: str, box: str, config: "MathVerifyConfig") -> bool:
    latex = LatexExtractionConfig()
    latex = replace(latex, normalization_config=replace(latex.normalization_config, units=config.units))
    gold = _parse_answer(reference, latex)
    if gold is None:
        return False
    prediction = _parse_answer(box, latex)
    if prediction is None:
        return False
    gold_name, left = gold
    prediction_name, right = prediction
    if gold_name and prediction_name and gold_name != prediction_name:
        return False
    if isinstance(left, (FiniteSet, Tuple)) or isinstance(right, (FiniteSet, Tuple)):
        return left == right
    if isinstance(left, Rational) and isinstance(right, Rational):
        if left != right and ("." in reference or "." in box):
            # Distinct rationals differ by at least 1/(q1*q2). More places cannot make them round alike.
            places = min(config.precision, int(left.q).bit_length() + int(right.q).bit_length())
            left, right = Rational(round(Fraction(left), places)), Rational(round(Fraction(right), places))
    elif left != right:
        # Math-Verify can round tiny nonzero symbolic offsets to zero even at high precision.
        return False
    return verify(left, right, float_rounding=config.precision, timeout_seconds=0) and (
        not config.bidirectional or verify(right, left, float_rounding=config.precision, timeout_seconds=0)
    )


def _score_box(reference: str, box: str, config: "MathVerifyConfig", parent_pid: int) -> bool:
    if sys.platform == "linux":
        # A replaced server worker's fork must release its inherited listening socket.
        pr_set_pdeathsig = 1
        prctl = ctypes.CDLL(None).prctl
        prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        if prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError("Could not register the scorer's parent-death signal")
        if os.getppid() != parent_pid:  # The parent died before prctl ran.
            return False
    else:
        # Without Linux's parent-death signal, bound an orphan even if its server worker is killed.
        # Let a live parent's normal timeout report first. Library calls leave this alarm alone.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL, config.timeout_seconds + 1)
    # The forked verifier must not run uvicorn's shutdown handlers.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)
    # The worker owns the deadline. Avoid the library's per-process warning about disabled internal alarms.
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        try:
            return _matches(reference, box, config)
        except ValueError:
            return False


async def _reap(pid: int) -> None:
    """Collect the killed child before releasing its slot, even if the host cannot start a thread."""
    cancelled = False
    while os.waitpid(pid, os.WNOHANG)[0] == 0:
        try:
            await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


class BoxedMathVerifier:
    def __init__(self, config: "MathVerifyConfig", server_name: str):
        self.config = config
        self.slots = MeteredSemaphore(
            config.max_concurrency,
            component="resources_server",
            limit_name="library_verifier",
            server_name=server_name,
        )

    async def matches(self, reference: str, box: str) -> bool:
        async with resources_operation(self.slots, operation="verify", executor="subprocess") as operation:
            # Own both ends before forking so a failed launch releases every descriptor.
            loop = asyncio.get_running_loop()
            receiver, sender = socket.socketpair()
            with receiver, sender:
                parent_pid = os.getpid()
                pid = os.fork()
                if pid == 0:
                    try:
                        receiver.close()
                        matched = _score_box(reference, box, self.config, parent_pid)
                        sender.sendall(b"1" if matched else b"0")
                    finally:
                        os._exit(0)  # Never resume the server event loop in the child.
                try:
                    sender.close()
                    receiver.setblocking(False)
                    result = await asyncio.wait_for(loop.sock_recv(receiver, 1), self.config.timeout_seconds)
                    if result:
                        return result == b"1"
                    operation.outcome = "crashed"
                except TimeoutError:
                    operation.outcome = "timeout"
                finally:
                    if os.waitpid(pid, os.WNOHANG)[0] == 0:
                        os.kill(pid, signal.SIGKILL)
                        await _reap(pid)
                return False
