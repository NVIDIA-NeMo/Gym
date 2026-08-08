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
