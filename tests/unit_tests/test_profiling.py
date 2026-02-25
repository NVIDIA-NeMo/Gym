from pathlib import Path

from pydantic import ValidationError
from pytest import raises

from nemo_gym.profiling import Profiler


class TestProfiling:
    def test_sanity(self, tmp_path: Path) -> None:
        profiler = Profiler(name="test_name", base_profile_dir=tmp_path / "profile")
        profiler.start()
        profiler.stop()

    def test_profiler_errors_on_invalid_name(self, tmp_path: Path) -> None:
        with raises(ValidationError, match="Spaces are not allowed"):
            Profiler(name="test name", base_profile_dir=tmp_path / "profile")
