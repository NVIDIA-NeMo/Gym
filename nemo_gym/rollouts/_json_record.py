# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare reordered historical Gym JSON records without loading their captured observations."""

import codecs
import hashlib
import re
import sqlite3
from contextlib import closing
from itertools import count
from pathlib import Path

import orjson

from nemo_gym.config.config_types import ConfigError


_STRING_BOUNDARY = re.compile(rb'["\\\x00-\x1f]')
_SCALAR = re.compile(rb"[a-zA-Z0-9.+-]+")


def json_record_digest(path: Path, offset: int) -> bytes:
    """Hash one Gym-written JSON value. Array order matters and object keys can arrive in any order.

    Most replays have identical bytes. This slower path only handles differing serializations of large records.
    Strings are read by chunk and arrays by element. An unnamed temporary database orders object members on disk,
    so a large string, a long list of model calls, or a wide object need not fit in memory.
    """
    with path.open("rb") as stream, closing(sqlite3.connect("")) as members:
        stream.seek(offset)
        members.execute(
            "CREATE TABLE member (obj INTEGER, key BLOB, value BLOB, PRIMARY KEY (obj, key)) WITHOUT ROWID"
        )
        objects = count()

        def token() -> bytes:
            char = stream.read(1)
            while char in (b" ", b"\t", b"\r"):
                char = stream.read(1)
            return char

        def string_digest() -> bytes:
            digest = hashlib.sha256(b'"')
            utf8 = codecs.getincrementaldecoder("utf-8")()
            while chunk := stream.peek(1):
                boundary = _STRING_BOUNDARY.search(chunk)
                fragment = stream.read(boundary.start() if boundary else len(chunk))
                utf8.decode(fragment, final=boundary is not None)
                digest.update(fragment)
                if boundary is None:
                    continue
                char = stream.read(1)
                if char == b'"':
                    return digest.digest()
                if char != b"\\":
                    break
                escape = stream.read(1)
                if escape == b"u":
                    code = stream.read(4)
                    escape += code
                    if 0xD800 <= int(code, 16) <= 0xDBFF:
                        escape += stream.read(6)  # the low surrogate
                decoded = orjson.loads(b'"\\' + escape + b'"')
                digest.update(decoded.encode())
            raise ValueError("Invalid JSON string in a saved rollout")

        def value(first: bytes) -> bytes:
            if first == b'"':
                return string_digest()
            if first == b"{":
                obj = next(objects)
                char = token()
                while char == b'"':
                    key = string_digest()
                    if token() != b":":
                        raise ValueError("Expected a JSON object member")
                    members.execute("INSERT OR REPLACE INTO member VALUES (?, ?, ?)", (obj, key, value(token())))
                    char = token()
                    if char != b",":
                        break
                    char = token()
                    if char != b'"':
                        raise ValueError("Expected another JSON object member")
                if char != b"}":
                    raise ValueError("Invalid JSON object in a saved rollout")
                digest = hashlib.sha256(first)
                for key, item in members.execute("SELECT key, value FROM member WHERE obj=? ORDER BY key", (obj,)):
                    digest.update(key + item)
                members.execute("DELETE FROM member WHERE obj=?", (obj,))
                return digest.digest()
            if first == b"[":
                digest = hashlib.sha256(first)
                char = token()
                if char != b"]":
                    while True:
                        digest.update(value(char))
                        char = token()
                        if char != b",":
                            break
                        char = token()
                if char != b"]":
                    raise ValueError("Invalid JSON array in a saved rollout")
                return digest.digest()
            # Gym writes JSON numbers as finite doubles or 64-bit integers. Match ordinary decoded numeric equality.
            scalar = first
            while match := _SCALAR.match(stream.peek(1)):
                scalar += stream.read(match.end())
                if len(scalar) > 64:
                    raise ValueError("Invalid scalar in a saved rollout")
            parsed = orjson.loads(scalar)
            if isinstance(parsed, (int, float)):
                return hashlib.sha256(str(parsed.as_integer_ratio()).encode()).digest()
            return hashlib.sha256(scalar).digest()

        try:
            result = value(token())
            if token() not in (b"", b"\n"):
                raise ValueError("Extra data after the JSON record")
            return result
        except (ValueError, RecursionError) as error:
            raise ConfigError(f"Invalid JSON at byte {offset} of {path}.") from error
