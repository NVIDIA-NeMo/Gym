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
import datetime
import json
import threading
import time
from collections.abc import Iterator

import orjson
import pytest
import uvicorn
from fastapi import Body, FastAPI

from nemo_gym.client.http_client import get_response_json, request
from nemo_gym.utils.json_codec import dumps_json_bytes, loads_json


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
    assert dumps_json_bytes(value, option=option) == orjson.dumps(value, option=option)


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

    data = dumps_json_bytes(value)

    assert data == encoded.encode("utf-8")
    data.decode("utf-8")  # valid UTF-8, even with a lone surrogate in the value
    assert loads_json(data) == decoded
    assert json.loads(data) == decoded


def test_fallback_honors_sort_keys_indent_and_newline() -> None:
    value = {"b": WIDE, "a": {"d": 1, "c": [2]}}
    option = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE

    data = dumps_json_bytes(value, option=option)

    narrow = {"b": 1, "a": {"d": 1, "c": [2]}}
    assert data == orjson.dumps(narrow, option=option).replace(b'"b": 1', b'"b": ' + str(WIDE).encode())
    assert loads_json(data) == value


def test_fallback_writes_other_types_as_orjson_does() -> None:
    at = datetime.datetime(2026, 11, 6, 12, 0)
    assert dumps_json_bytes({"n": WIDE, "at": at}) == b'{"n":1180591620717411303424,"at":"2026-11-06T12:00:00"}'
    assert dumps_json_bytes({"n": WIDE, "x": object()}, default=lambda _: "custom") == (
        b'{"n":1180591620717411303424,"x":"custom"}'
    )


def test_nesting_deeper_than_orjson_allows_round_trips() -> None:
    value: object = WIDE
    for _ in range(300):
        value = [value]

    assert loads_json(dumps_json_bytes(value)) == value


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
        dumps_json_bytes(value)


def test_an_option_the_fallback_cannot_honor_raises_instead_of_dropping_it() -> None:
    with pytest.raises(orjson.JSONEncodeError, match="64-bit"):
        dumps_json_bytes({"n": WIDE}, option=orjson.OPT_NON_STR_KEYS)


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


@pytest.mark.parametrize("data", [None, 1, {"a": 1}])
def test_input_of_the_wrong_type_raises_orjsons_decode_error(data: object) -> None:
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
