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
"""The model-call hook host: loading, calling with the right arguments, and isolating failures."""

import logging
import textwrap

import pytest

from nemo_gym.telemetry import model_call
from nemo_gym.telemetry.model_call import load_hook, model_call_spans_enabled, record_model_call_spans
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import requires_lens


pytestmark = requires_lens

#: Epoch microseconds; the values only need to be ordered.
T0 = 1_700_000_000_000_000

HOOK_SOURCE = textwrap.dedent(
    """
    CALLS = []

    def hook(engine_response, *, span, emit_span, sent_us, received_us, replica_idx):
        CALLS.append((sent_us, received_us, replica_idx))
        span.set_attributes({"my.tokens": engine_response["tokens"], "my.replica": replica_idx})
        emit_span("decode", sent_us + 10, sent_us + 100, {"my.class": "compute"})
        emit_span("bare", sent_us + 100, sent_us + 105)

    def raises(engine_response, **call):
        raise RuntimeError("boom")

    not_callable = 3
    """
)


@pytest.fixture(autouse=True)
def _fresh_hooks():
    model_call._hooks.clear()
    yield
    model_call._hooks.clear()


@pytest.fixture
def hook_file(tmp_path):
    path = tmp_path / "my_hook.py"
    path.write_text(HOOK_SOURCE)
    return str(path)


@pytest.fixture
def spans(monkeypatch):
    """Enable the groups and capture finished spans with an in-memory exporter."""
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    set_enabled_span_groups(GymSpanGroup.resolve("all"))
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(model_call, "_tracer", lambda: tracer)
    yield tracer, exporter.get_finished_spans
    set_enabled_span_groups(frozenset())


def _response(**overrides):
    return {"tokens": 21, **overrides}


def _record(tracer, response, **kwargs):
    kwargs.setdefault("started_us", T0 - 5)
    kwargs.setdefault("sent_us", T0)
    kwargs.setdefault("received_us", T0 + 160)
    kwargs.setdefault("finished_us", T0 + 170)
    with tracer.start_as_current_span("gym.model.responses"):
        record_model_call_spans(response, **kwargs)


def _by_name(get_finished_spans):
    return {span.name: span for span in get_finished_spans()}


def test_without_a_hook_only_the_model_servers_own_spans_are_recorded(spans):
    tracer, finished = spans
    _record(tracer, _response(), replica_idx=3)

    by_name = _by_name(finished)
    assert set(by_name) == {"gym.model.responses", "gym.model.prepare", "gym.model.postprocess"}
    assert dict(by_name["gym.model.responses"].attributes) == {}
    prepare, postprocess = by_name["gym.model.prepare"], by_name["gym.model.postprocess"]
    assert (prepare.start_time, prepare.end_time) == ((T0 - 5) * 1_000, T0 * 1_000)
    assert (postprocess.start_time, postprocess.end_time) == ((T0 + 160) * 1_000, (T0 + 170) * 1_000)
    parent = by_name["gym.model.responses"]
    assert prepare.parent.span_id == postprocess.parent.span_id == parent.context.span_id


def test_the_rollout_and_session_ids_go_on_the_model_call_span(spans):
    tracer, finished = spans
    _record(tracer, _response(), rollout_id="3-1", session_id="session-a")

    parent = _by_name(finished)["gym.model.responses"]
    assert dict(parent.attributes) == {"nemo.gym.rollout.id": "3-1", "nemo.gym.session.id": "session-a"}


def test_an_id_the_request_did_not_carry_is_left_off(spans):
    tracer, finished = spans
    _record(tracer, _response(), rollout_id=None, session_id="session-a")

    assert dict(_by_name(finished)["gym.model.responses"].attributes) == {"nemo.gym.session.id": "session-a"}


def test_a_tokenize_call_splits_the_post_processing(spans):
    tracer, finished = spans
    _record(tracer, _response(), tokenize_start_us=T0 + 161, tokenize_end_us=T0 + 168)

    by_name = _by_name(finished)
    assert set(by_name) == {
        "gym.model.responses",
        "gym.model.prepare",
        "gym.model.postprocess",
        "gym.model.tokenize",
        "gym.model.finalize",
    }
    times = {
        name: (by_name[f"gym.model.{name}"].start_time // 1_000, by_name[f"gym.model.{name}"].end_time // 1_000)
        for name in ("postprocess", "tokenize", "finalize")
    }
    assert times == {
        "postprocess": (T0 + 160, T0 + 161),
        "tokenize": (T0 + 161, T0 + 168),
        "finalize": (T0 + 168, T0 + 170),
    }


def test_a_hook_sets_attributes_and_draws_spans_exactly_as_it_chooses(spans, hook_file):
    tracer, finished = spans
    _record(tracer, _response(), hook=f"{hook_file}:hook", replica_idx=3)

    by_name = _by_name(finished)
    parent = by_name["gym.model.responses"]
    assert dict(parent.attributes) == {"my.tokens": 21, "my.replica": 3}  # Gym adds no prefix and no attribute
    assert set(by_name) == {
        "gym.model.responses",
        "gym.model.prepare",
        "gym.model.llm_execute",
        "gym.model.postprocess",
        "gym.model.decode",
        "gym.model.bare",
    }
    decode = by_name["gym.model.decode"]
    assert dict(decode.attributes) == {"my.class": "compute"}
    assert (decode.start_time, decode.end_time) == ((T0 + 10) * 1_000, (T0 + 100) * 1_000)
    execute = by_name["gym.model.llm_execute"]
    assert (execute.start_time, execute.end_time) == (T0 * 1_000, (T0 + 160) * 1_000)
    assert execute.parent.span_id == parent.context.span_id
    assert decode.parent.span_id == execute.context.span_id  # the hook's spans are nested under llm_execute
    assert dict(by_name["gym.model.bare"].attributes) == {}


def test_the_hook_gets_the_engine_boundaries_and_is_loaded_once(spans, hook_file, monkeypatch):
    tracer, _ = spans
    loads = []
    real = model_call.load_hook
    monkeypatch.setattr(model_call, "load_hook", lambda spec: loads.append(spec) or real(spec))
    for _ in range(3):
        _record(tracer, _response(), hook=f"{hook_file}:hook", replica_idx=3)

    assert loads == [f"{hook_file}:hook"]
    assert model_call._hooks[f"{hook_file}:hook"].__globals__["CALLS"] == [(T0, T0 + 160, 3)] * 3


def test_a_hook_that_raises_is_logged_once_and_skipped(spans, hook_file, caplog):
    tracer, finished = spans
    with caplog.at_level(logging.WARNING, logger=model_call.logger.name):
        _record(tracer, _response(), hook=f"{hook_file}:raises")
        _record(tracer, _response(), hook=f"{hook_file}:raises")

    assert [span.name for span in finished()].count("gym.model.responses") == 2
    assert sum("failed; skipping" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("spec", ["no_colon", "/nope/missing.py:hook", "{file}:missing", "{file}:not_callable"])
def test_an_unusable_hook_is_logged_once_and_the_call_goes_on(spans, hook_file, caplog, spec):
    tracer, finished = spans
    with caplog.at_level(logging.WARNING, logger=model_call.logger.name):
        _record(tracer, _response(), hook=spec.format(file=hook_file))
        _record(tracer, _response(), hook=spec.format(file=hook_file))

    assert [span.name for span in finished()].count("gym.model.responses") == 2
    assert sum("could not be loaded" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("spec", ["hook", "file.txt:hook", "/x/file.py:not-an-identifier", "/x/file.py:"])
def test_load_hook_rejects_a_malformed_spec(spec):
    with pytest.raises(ValueError, match="llm_span_breakdown_hook"):
        load_hook(spec)


def test_nothing_is_recorded_or_loaded_when_the_group_is_off(hook_file):
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(frozenset())
    assert not model_call_spans_enabled()
    record_model_call_spans(
        _response(), started_us=T0, sent_us=T0, received_us=T0, finished_us=T0, hook=f"{hook_file}:hook"
    )

    assert model_call._hooks == {}


def test_model_call_spans_enabled_follows_the_model_call_group(spans):
    assert model_call_spans_enabled()
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(GymSpanGroup.resolve("server"))
    assert not model_call_spans_enabled()
