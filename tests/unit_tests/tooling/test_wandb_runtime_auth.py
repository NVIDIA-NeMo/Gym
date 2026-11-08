# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential loading uses synthetic SDK responses, never the user's saved login."""

import sys
from types import ModuleType

import pytest
from scripts import with_wandb_auth


def test_inherited_key_skips_sdk_and_forces_online(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-inherited-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setitem(sys.modules, "wandb", None)
    result = with_wandb_auth.runtime_environment()
    assert result["WANDB_API_KEY"] == "synthetic-inherited-key"
    assert result["WANDB_MODE"] == "online" and "WANDB_DISABLED" not in result


@pytest.mark.parametrize("failure", [False, True, "missing"])
def test_saved_lookup_passes_key_only_in_child_environment(monkeypatch, capsys, failure):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
    secret = "synthetic-saved-key"
    sdk = ModuleType("wandb.sdk.lib.wbauth")

    def lookup(*, host):
        assert host == "https://api.wandb.ai"
        print(secret)
        if failure == "missing":
            return None
        if failure:
            raise ValueError(secret)
        return secret

    sdk.read_netrc_auth = lookup
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib.wbauth", sdk)
    monkeypatch.setattr(sys, "argv", ["with_wandb_auth.py", "--", "/bin/true", "safe-argument"])
    launched = []
    monkeypatch.setattr(with_wandb_auth.os, "execvpe", lambda *args: launched.append(args))
    if failure:
        with pytest.raises(SystemExit) as error:
            with_wandb_auth.main()
        assert error.value.code == 1 and not launched
    else:
        with_wandb_auth.main()
        executable, argv, environment = launched[0]
        assert executable == "/bin/true" and argv == ["/bin/true", "safe-argument"]
        assert environment["WANDB_API_KEY"] == secret
    output = capsys.readouterr()
    assert secret not in output.out + output.err
