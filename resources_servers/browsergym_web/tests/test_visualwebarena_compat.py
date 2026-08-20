# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from resources_servers.browsergym_web import visualwebarena_compat


def test_evaluator_model_override_preserves_prompt_and_generation_options(monkeypatch):
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "correct"

    provider = SimpleNamespace(generate_from_openai_chat_completion=original)
    helpers = SimpleNamespace(generate_from_openai_chat_completion=original)

    def fake_import(name):
        if name.endswith("openai_utils"):
            return provider
        if name.endswith("helper_functions"):
            return helpers
        raise AssertionError(name)

    monkeypatch.setattr(visualwebarena_compat.importlib, "import_module", fake_import)
    visualwebarena_compat.configure_evaluator_model("azure/anthropic/claude-opus-4-7")

    answer = helpers.generate_from_openai_chat_completion(
        messages=[{"role": "user", "content": "grade this"}],
        model="gpt-4-1106-preview",
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    )

    assert answer == "correct"
    assert calls == [
        (
            (),
            {
                "messages": [{"role": "user", "content": "grade this"}],
                "model": "azure/anthropic/claude-opus-4-7",
                "temperature": 0,
                "max_tokens": 768,
                "top_p": 1.0,
                "context_length": 0,
            },
        )
    ]


def test_webarena_uses_its_own_provider_and_helper_modules(monkeypatch):
    calls = []
    imports = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "same"

    provider = SimpleNamespace(generate_from_openai_chat_completion=original)
    helpers = SimpleNamespace(generate_from_openai_chat_completion=original)

    def fake_import(name):
        imports.append(name)
        if name == "webarena.llms.providers.openai_utils":
            return provider
        if name == "webarena.evaluation_harness.helper_functions":
            return helpers
        raise AssertionError(name)

    monkeypatch.setattr(visualwebarena_compat.importlib, "import_module", fake_import)
    visualwebarena_compat.configure_webarena_evaluator_model("local-judge")

    assert helpers.generate_from_openai_chat_completion([], "gpt-4-1106-preview", 0, 768, 1.0, 0) == "same"
    assert imports == [
        "webarena.llms.providers.openai_utils",
        "webarena.evaluation_harness.helper_functions",
    ]
    assert calls[0][0][1] == "local-judge"


def test_evaluator_environment_supports_openai_v0_and_v1(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    visualwebarena_compat.configure_evaluator_environment(
        api_key="test-only",
        base_url="http://judge.test/v1",
    )

    assert visualwebarena_compat.os.environ["OPENAI_API_KEY"] == "test-only"
    assert visualwebarena_compat.os.environ["OPENAI_BASE_URL"] == "http://judge.test/v1"
    assert visualwebarena_compat.os.environ["OPENAI_API_BASE"] == "http://judge.test/v1"


def test_rule_only_import_environment_restores_process_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://existing.test/v1")

    with visualwebarena_compat.rule_only_evaluator_import_environment(
        base_url="http://temporary.test/v1",
    ):
        assert visualwebarena_compat.os.environ["OPENAI_API_KEY"] == "unused-for-rule-only-evaluator"
        assert visualwebarena_compat.os.environ["OPENAI_BASE_URL"] == "http://temporary.test/v1"
        assert visualwebarena_compat.os.environ["OPENAI_API_BASE"] == "http://temporary.test/v1"

    assert visualwebarena_compat.os.environ["OPENAI_API_KEY"] == "existing-key"
    assert "OPENAI_BASE_URL" not in visualwebarena_compat.os.environ
    assert visualwebarena_compat.os.environ["OPENAI_API_BASE"] == "http://existing.test/v1"
