from argparse import Namespace
from types import SimpleNamespace

import pytest

from nemo_gym.cli.main import _eval_resubmit
from nemo_gym.orchestration import submit


@pytest.mark.parametrize(
    "requested,status,expected",
    [(True, "SUCCESS", 0), (True, "FAILED", 1), (True, "UNKNOWN", 1), (False, "FAILED", 0)],
)
def test_requested_gate_controls_exit(monkeypatch, requested, status, expected):
    outcome = SimpleNamespace(ok=True, gate_task={"status": status}, model_dump_json=lambda **kw: "{}")
    monkeypatch.setattr(submit, "recover", lambda *a, **kw: outcome)
    args = Namespace(
        gym_job_id="saved", benchmark="judge", recover=True, score_key='["reward"]' if requested else None
    )
    if expected:
        with pytest.raises(SystemExit) as error:
            _eval_resubmit(args, [])
        assert error.value.code == expected
    else:
        _eval_resubmit(args, [])
