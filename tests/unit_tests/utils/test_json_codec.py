# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import dataclasses
import datetime
import enum
import json
import threading
import time
from collections.abc import Iterator

import numpy as np
import orjson
import pytest
import uvicorn
from fastapi import Body, FastAPI

from nemo_gym.client.http_client import get_response_json, request
from nemo_gym.utils import json_codec
from nemo_gym.utils.json_codec import dumps_json, loads_json


WIDE = 2**70
NAN = float("nan")


@pytest.mark.parametrize(
    ("value", "option"),
    [
        ({"text": "café", "token_ids": [1, 2, 3], "logprob": -0.000321865081787109375}, None),
        ({"b": 1, "a": [2**64 - 1, -(2**63)], "nan": NAN, "inf": float("-inf")}, orjson.OPT_SORT_KEYS),
        ({"nested": {"z": None, "a": True}}, orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
        ({"at": datetime.datetime(2026, 11, 6, 12, 0)}, orjson.OPT_APPEND_NEWLINE),
    ],
)
def test_values_orjson_accepts_keep_their_orjson_bytes(value: object, option: int | None) -> None:
    assert dumps_json(value, option=option) == orjson.dumps(value, option=option)


@pytest.mark.parametrize(
    ("value", "encoded", "decoded"),
    [
        pytest.param(
            {"n": WIDE, "m": -WIDE, "note": "ü"},
            '{"n":1180591620717411303424,"m":-1180591620717411303424,"note":"ü"}',
            {"n": WIDE, "m": -WIDE, "note": "ü"},
            id="wide-integer",
        ),
        pytest.param(
            {"s": "\ud83d", "note": "ü"},
            '{"s":"\\ud83d","note":"\\u00fc"}',
            {"s": "\ud83d", "note": "ü"},
            id="lone-surrogate",
        ),
        # orjson writes NaN as null; the fallback does too, rather than writing invalid JSON or failing only
        # because the same body also holds a wide integer.
        pytest.param(
            {"n": WIDE, "f": NAN, "g": [float("inf")]},
            '{"n":1180591620717411303424,"f":null,"g":[null]}',
            {"n": WIDE, "f": None, "g": [None]},
            id="nan-with-wide-integer",
        ),
    ],
)
def test_values_orjson_rejects_round_trip_exactly(value: object, encoded: str, decoded: object) -> None:
    with pytest.raises(orjson.JSONEncodeError):
        orjson.dumps(value)

    data = dumps_json(value)

    assert data == encoded.encode("utf-8")
    data.decode("utf-8")  # valid UTF-8, even with a lone surrogate in the value
    assert repr(loads_json(data)) == repr(decoded)  # == alone accepts the float 2**70
    assert json.loads(data) == decoded


def test_fallback_honors_sort_keys_indent_and_newline() -> None:
    value = {"b": WIDE, "a": {"d": 1, "c": [2]}}
    option = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE

    data = dumps_json(value, option=option)

    narrow = {"b": 1, "a": {"d": 1, "c": [2]}}
    assert data == orjson.dumps(narrow, option=option).replace(b'"b": 1', b'"b": ' + str(WIDE).encode())
    decoded = loads_json(data)
    assert decoded == value
    assert type(decoded["b"]) is int  # == alone accepts the float 2**70


def test_fallback_writes_other_types_as_orjson_does() -> None:
    at = datetime.datetime(2026, 11, 6, 12, 0)
    assert dumps_json({"n": WIDE, "at": at}) == b'{"n":1180591620717411303424,"at":"2026-11-06T12:00:00"}'
    assert dumps_json({"n": WIDE, "x": object()}, default=lambda _: "custom") == (
        b'{"n":1180591620717411303424,"x":"custom"}'
    )


@dataclasses.dataclass
class _Call:
    name: str
    arguments: dict
    _private: int = 7


class _UndecoratedCall(_Call):
    pass


class _Size(enum.Enum):
    WIDE = 2**70


def test_dataclasses_and_enums_holding_a_wide_integer_are_written_not_passed_to_default() -> None:
    narrow = _Call("f", {"n": 1})
    assert dumps_json(narrow, default=str) == orjson.dumps(narrow)

    # Fields named with a leading underscore are left out, as orjson leaves them out.
    assert dumps_json(_Call("f", {"n": WIDE}), default=str) == b'{"name":"f","arguments":{"n":1180591620717411303424}}'
    assert dumps_json({"size": _Size.WIDE}, default=str) == b'{"size":1180591620717411303424}'

    # orjson refuses an undecorated subclass of a dataclass, so the fallback does too, or calls default.
    undecorated = _UndecoratedCall("f", {"n": 1})
    with pytest.raises(orjson.JSONEncodeError):
        orjson.dumps(undecorated)
    with pytest.raises(orjson.JSONEncodeError):
        dumps_json(undecorated)
    assert dumps_json({"n": WIDE, "v": undecorated}, default=lambda _: "default") == (
        b'{"n":1180591620717411303424,"v":"default"}'
    )


def test_nesting_deeper_than_orjson_allows_round_trips() -> None:
    value: object = WIDE
    for _ in range(300):
        value = [value]

    assert repr(loads_json(dumps_json(value))) == repr(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({1: "a", "1": "b"}, id="int-key"),
        pytest.param({"n": WIDE, "inner": {1: "a"}}, id="int-key-beside-wide-integer"),
        pytest.param({"n": WIDE, "s": {1, 2}}, id="set"),
    ],
)
def test_values_neither_encoder_writes_exactly_raise_orjsons_error(value: object) -> None:
    with pytest.raises(orjson.JSONEncodeError):
        dumps_json(value)


_Pair = collections.namedtuple("_Pair", "a b")


class _Float(float):
    pass


@pytest.mark.parametrize(
    "value", [_Pair(1, 2), _Float(1.5), np.float64(1.5)], ids=["namedtuple", "float-subclass", "numpy-float"]
)
def test_types_orjson_refuses_are_refused_with_or_without_a_wide_integer(value: object) -> None:
    with pytest.raises(orjson.JSONEncodeError):
        orjson.dumps(value)

    with pytest.raises(orjson.JSONEncodeError):
        dumps_json(value)
    with pytest.raises(orjson.JSONEncodeError):
        dumps_json({"n": WIDE, "v": value})
    assert dumps_json({"n": WIDE, "v": value}, default=repr) == (
        b'{"n":1180591620717411303424,"v":"%s"}' % repr(value).encode()
    )


def test_an_option_the_fallback_cannot_honor_raises_instead_of_dropping_it() -> None:
    with pytest.raises(orjson.JSONEncodeError, match="64-bit"):
        dumps_json({"n": WIDE}, option=orjson.OPT_NON_STR_KEYS)


@pytest.mark.parametrize(
    "value",
    [
        2**64 - 1,
        2**64,
        -(2**63),
        -(2**63) - 1,
        WIDE,
        -WIDE,
        10**40,
        [1, WIDE, {"k": -(10**25)}],
    ],
)
def test_integers_decode_exactly(value: object) -> None:
    encoded = json.dumps(value).encode()

    decoded = loads_json(encoded)

    assert decoded == value
    assert json.dumps(decoded) == json.dumps(value)  # still ints, not floats that compare equal
    assert loads_json(encoded.decode()) == value
    assert loads_json(bytearray(encoded)) == value
    assert loads_json(memoryview(encoded)) == value


@pytest.mark.parametrize(
    "data",
    [
        # Long digit runs that are not integers: fractions, a float written with its integer digits, and text.
        b"[-0.00032186508178710937500, 0.12345678901234567890123]",
        b"[123456789012345678901.5, 1e25, 12345678901234567890123e-3]",
        b'{"id": "12345678901234567890123", "trace": "x-99999999999999999999"}',
    ],
)
def test_long_digit_runs_that_are_not_wide_integers_decode_as_orjson_does(data: bytes) -> None:
    assert loads_json(data) == orjson.loads(data)


@pytest.mark.parametrize(
    "data",
    [
        b'{"id": "msg_12345678901234567890123", "path": "/tmp/run_1762400000123456789/out"}',
        b'{"sha": "0000000000000000000000000000000000000000", "logprob": -0.000321865081787109375}',
    ],
)
def test_long_digit_runs_inside_strings_stay_on_orjson(data: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_fallback(*args: object) -> object:
        raise AssertionError("parsed with the standard library")

    monkeypatch.setattr(json_codec, "_loads_stdlib", no_fallback)

    assert loads_json(data) == orjson.loads(data)


def test_an_escaped_lone_surrogate_decodes() -> None:
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(b'{"s": "\\ud800"}')

    assert loads_json(b'{"s": "\\ud800", "pair": "\\ud83d\\ude00"}') == {"s": "\ud800", "pair": "\U0001f600"}


@pytest.mark.parametrize(
    "data",
    [
        b"[NaN]",
        b"[-Infinity]",
        b"[1e400]",
        b"[" + b"9" * 5000 + b"]",
        b'{"s": "\\ud800", "f": NaN}',
        b"\xef\xbb\xbf[1]",
        b'"\xff"',
        b"{",
    ],
)
def test_bodies_orjson_rejects_still_raise_its_decode_error(data: bytes) -> None:
    with pytest.raises(orjson.JSONDecodeError):
        loads_json(data)


@pytest.mark.parametrize("data", [None, 10**12, {"a": 1}, list(b"[1]"), memoryview(b"[x1x,x2x]x")[0::2]])
def test_input_of_the_wrong_type_raises_orjsons_decode_error(data: object) -> None:
    # ``bytes()`` would decode a list of byte values as JSON, or allocate an integer's worth of zeros first.
    with pytest.raises(orjson.JSONDecodeError, match="Input must be bytes"):
        loads_json(data)  # type: ignore[arg-type]


def test_a_body_both_parsers_reject_raises_orjsons_error() -> None:
    with pytest.raises(orjson.JSONDecodeError) as raised:
        loads_json(b'{"a": 1,}')

    with pytest.raises(orjson.JSONDecodeError) as direct:
        orjson.loads(b'{"a": 1,}')
    assert str(raised.value) == str(direct.value)


def _tool_app() -> FastAPI:
    """A tool endpoint the way a resources server exposes one: FastAPI parses the arguments and returns JSON."""
    app = FastAPI()

    @app.post("/echo")
    async def _echo(body: dict = Body()) -> dict:
        return {"echo": body}

    @app.post("/describe")
    async def _describe(body: dict = Body()) -> dict:
        return {"seen": {key: [type(value).__name__, ascii(value)] for key, value in body.items()}}

    return app


@pytest.fixture
def tool_server_url() -> Iterator[str]:
    server = uvicorn.Server(uvicorn.Config(_tool_app(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the test server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


async def test_tool_arguments_cross_a_resources_server_and_back_exactly(tool_server_url: str) -> None:
    arguments = {"n": WIDE, "m": -(2**63) - 1, "f": NAN, "text": "ü"}

    response = await request("POST", f"{tool_server_url}/echo", json=arguments)

    assert response.status == 200
    reply = await get_response_json(response)
    assert reply == {"echo": {"n": WIDE, "m": -(2**63) - 1, "f": None, "text": "ü"}}
    assert type(reply["echo"]["n"]) is int and type(reply["echo"]["m"]) is int


async def test_a_lone_surrogate_reaches_the_resources_server_intact(tool_server_url: str) -> None:
    response = await request("POST", f"{tool_server_url}/describe", json={"s": "a\ud800b", "n": WIDE})

    assert response.status == 200
    assert await get_response_json(response) == {"seen": {"s": ["str", ascii("a\ud800b")], "n": ["int", str(WIDE)]}}


@pytest.mark.parametrize(
    "data",
    [
        # Token IDs and logprobs as replies carry them: long fractions and integers below 19 digits.
        b'{"token_ids": [151643, 999999999999999999], "logprobs": [-0.00032186508178710937500, -1.5e-07]}',
        b"[0.12345678901234567890123, -999999999999999999]",
    ],
)
def test_common_bodies_never_take_the_standard_library_path(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    def no_fallback(*args: object) -> object:
        raise AssertionError("parsed with the standard library")

    monkeypatch.setattr(json_codec, "_loads_stdlib", no_fallback)

    assert loads_json(data) == orjson.loads(data)
    assert loads_json(data.decode()) == orjson.loads(data)


def test_an_invalid_body_holding_a_wide_integer_raises_orjsons_error() -> None:
    with pytest.raises(orjson.JSONDecodeError) as raised:
        loads_json(b"[1180591620717411303424,")

    assert type(raised.value) is orjson.JSONDecodeError


def test_a_str_holding_a_lone_surrogate_decodes() -> None:
    assert loads_json('{"s": "a\ud800b"}') == {"s": "a\ud800b"}


def test_nesting_deeper_than_the_standard_library_allows_raises_orjsons_error() -> None:
    with pytest.raises(orjson.JSONDecodeError):
        loads_json(b"[" * 100_000 + str(WIDE).encode() + b"]" * 100_000)


def test_nan_in_a_tuple_beside_a_wide_integer_is_written_as_null() -> None:
    assert dumps_json({"n": WIDE, "t": (NAN, 1)}) == b'{"n":1180591620717411303424,"t":[null,1]}'


class _Key(enum.StrEnum):
    A = "a"


@pytest.mark.parametrize("key", [_Key.A, type("_Str", (str,), {})("a")], ids=["str-enum", "str-subclass"])
def test_string_subclass_keys_are_refused_as_orjson_refuses_them(key: str) -> None:
    with pytest.raises(orjson.JSONEncodeError):
        orjson.dumps({key: 1})

    with pytest.raises(orjson.JSONEncodeError):
        dumps_json({key: 1, "n": WIDE})
