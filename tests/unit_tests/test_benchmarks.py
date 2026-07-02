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
from glob import glob
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from yaml import safe_load

from nemo_gym.cli.eval import _benchmark_domain, _fuzzy_matches, list_benchmarks, prepare_benchmark


def _mock_global_config(config: dict = None):
    """Return an OmegaConf config without CLI/file parsing."""
    return OmegaConf.create(config or {})


class TestListBenchmarks:
    def test_lists_found_benchmarks(self, capsys) -> None:
        with patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config()):
            list_benchmarks()
        assert "aime24" in capsys.readouterr().out

    def test_discovers_by_type_benchmark_not_filename(self, tmp_path, capsys) -> None:
        # Discovery is content-based (a `type: benchmark` dataset), not filename-based: any yaml that
        # declares such a dataset is a candidate (e.g. tau2's `configs/tau2.yaml`), and yamls that don't
        # are skipped — regardless of filename.
        (tmp_path / "standard").mkdir()
        (tmp_path / "standard" / "config.yaml").write_text("x:\n  datasets:\n  - type: benchmark\n")
        (tmp_path / "flavored" / "configs").mkdir(parents=True)
        (tmp_path / "flavored" / "configs" / "myflavor.yaml").write_text("x:\n  datasets:\n  - type: benchmark\n")
        (tmp_path / "notbench").mkdir()
        (tmp_path / "notbench" / "config.yaml").write_text("x:\n  prompt_config: hi.yaml\n")  # no benchmark dataset

        captured = {}

        def fake_load(paths):
            captured["paths"] = {str(p.relative_to(tmp_path)) for p in paths}
            return {}

        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.eval.BENCHMARKS_DIR", tmp_path),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", side_effect=fake_load),
        ):
            list_benchmarks()

        assert captured["paths"] == {"standard/config.yaml", "flavored/configs/myflavor.yaml"}

    def test_no_benchmarks(self, capsys) -> None:
        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config()),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", return_value={}),
        ):
            list_benchmarks()
        assert "No benchmarks found" in capsys.readouterr().out

    def test_json_output(self, capsys) -> None:
        import json

        bench = MagicMock(agent_name="my_agent", num_repeats=4)
        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config({"json": True})),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", return_value={"my_bench": bench}),
            patch("nemo_gym.cli.eval._benchmark_domain", return_value="math"),
        ):
            list_benchmarks()
        assert json.loads(capsys.readouterr().out) == [
            {"name": "my_bench", "agent_name": "my_agent", "domain": "math", "num_repeats": 4}
        ]

    def test_json_output_empty(self, capsys) -> None:
        import json

        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config({"json": True})),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", return_value={}),
        ):
            list_benchmarks()
        assert json.loads(capsys.readouterr().out) == []


class TestLoadBenchmarksFromConfigPaths:
    def test_skips_configs_that_fail_to_resolve_with_warning(self, capsys) -> None:
        # A candidate that still can't be resolved even with tolerance (e.g. a multi-benchmark suite) must
        # be skipped with a warning — not crash the whole listing, and not vanish silently.
        from nemo_gym.benchmarks import BenchmarkConfig, _load_benchmarks_from_config_paths

        good = MagicMock()
        good.name = "good_bench"

        # Listing must resolve tolerantly (it scans files with no runtime context), so it opts out of strict.
        def fake_from_config_path(path, *, strict=True):
            assert strict is False
            if Path(path).name == "bad.yaml":
                raise RuntimeError("cannot resolve without runtime values")
            return good

        with patch.object(BenchmarkConfig, "from_config_path", side_effect=fake_from_config_path):
            result = _load_benchmarks_from_config_paths([Path("bad.yaml"), Path("good.yaml")])

        assert set(result) == {"good_bench"}
        err = capsys.readouterr().err
        assert "Warning" in err and "bad.yaml" in err

    def test_every_repo_benchmark_appears_in_listing(self, capsys) -> None:
        # Every config that declares a `type: benchmark` dataset must surface as its own listing entry —
        # no silent drop from a name collision (the name-keyed dict is last-writer-wins) or a resolve
        # failure. Mirrors the content-based discovery in `list_benchmarks`.
        from nemo_gym.benchmarks import BENCHMARKS_DIR, _load_benchmarks_from_config_paths

        config_paths = [BENCHMARKS_DIR / p for p in glob("**/*.yaml", root_dir=BENCHMARKS_DIR, recursive=True)]
        config_paths = sorted(p for p in config_paths if "type: benchmark" in p.read_text(errors="ignore"))
        assert config_paths, "no benchmark configs discovered under BENCHMARKS_DIR"

        benchmarks = _load_benchmarks_from_config_paths(config_paths)

        assert len(benchmarks) == len(config_paths), (
            f"{len(config_paths)} benchmark config(s) discovered but only {len(benchmarks)} appear in the "
            f"listing — a duplicate dataset name or resolve failure is hiding at least one.\n"
            f"stderr:\n{capsys.readouterr().err}"
        )


class TestTolerantInterpolationParse:
    # Unset `???` values and unresolved `${...}` interpolations reference runtime-only values that aren't
    # needed to identify a benchmark; listing fills them with a placeholder so the config still resolves.
    def _resolve(self, d: dict):
        from nemo_gym.benchmarks import _parse_no_environment_tolerating_unset_values

        return _parse_no_environment_tolerating_unset_values(OmegaConf.create(d))

    @property
    def _placeholder(self) -> str:
        from nemo_gym.benchmarks import _UNSET_VALUE_PLACEHOLDER

        return _UNSET_VALUE_PLACEHOLDER

    def test_single_interpolation(self) -> None:
        resolved = self._resolve({"foo": "${bar}"})
        assert resolved["foo"] == self._placeholder

    def test_single_missing_value(self) -> None:
        resolved = self._resolve({"foo": "???"})
        assert resolved["foo"] == self._placeholder

    def test_mix(self) -> None:
        # A mix across nested dicts: resolvable literals (incl. nested) pass through untouched, while an
        # undefined `${...}` interpolation and unset `???` values (incl. nested) are filled with the
        # placeholder.
        resolved = self._resolve(
            {
                "name": "my_bench",
                "num_repeats": 3,
                "api_key": "${some_api_key}",
                "server": {
                    "endpoint": "https://example.com",
                    "nested": {
                        "enabled": True,
                        "token": "???",
                    },
                },
            }
        )
        # Correct key-value pairs are unmodified.
        assert resolved["name"] == "my_bench"
        assert resolved["num_repeats"] == 3
        assert resolved["server"]["endpoint"] == "https://example.com"
        assert resolved["server"]["nested"]["enabled"] is True
        # Undefined `${...}` and unset `???` values are filled.
        assert resolved["api_key"] == self._placeholder
        assert resolved["server"]["nested"]["token"] == self._placeholder

    def test_does_not_mutate_input(self) -> None:
        from nemo_gym.benchmarks import _parse_no_environment_tolerating_unset_values

        cfg = OmegaConf.create({"foo": "???", "bar": "${baz}"})
        before = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
        _parse_no_environment_tolerating_unset_values(cfg)
        after = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
        assert after == before == {"foo": "???", "bar": "${baz}"}

    def test_strict_is_the_default_and_does_not_tolerate_unresolved_values(self) -> None:
        # The tolerance is listing-only: `from_initial_config_dict` defaults to strict, so other workflows
        # still get a hard error on an unresolved `${...}` rather than a silent placeholder.
        from omegaconf.errors import InterpolationKeyError

        from nemo_gym.benchmarks import BenchmarkConfig

        cfg = OmegaConf.create({"foo": "${runtime_only_value}"})
        with pytest.raises(InterpolationKeyError):
            BenchmarkConfig.from_initial_config_dict(path=Path("x.yaml"), initial_config_dict=cfg)

        # strict=False tolerates it (resolves, finds no benchmark dataset, returns None — no raise).
        tolerated = BenchmarkConfig.from_initial_config_dict(
            path=Path("x.yaml"), initial_config_dict=cfg, strict=False
        )
        assert tolerated is None


class TestFuzzyMatches:
    def test_substring_matches(self) -> None:
        assert _fuzzy_matches("math", "math_with_judge")

    def test_token_typo_matches(self) -> None:
        # `aimee` is a near-miss for the `aime` token in `aime24`.
        assert _fuzzy_matches("aimee", "aime24")

    def test_matches_against_agent_field(self) -> None:
        assert _fuzzy_matches("judge", "aime24", "math_with_judge_agent")

    def test_skips_empty_fields(self) -> None:
        assert not _fuzzy_matches("math", "", None)

    def test_no_match(self) -> None:
        assert not _fuzzy_matches("zzznomatch", "aime24", "math_with_judge")


class TestBenchmarkDomain:
    def test_resolves_domain_from_real_config(self) -> None:
        from nemo_gym.benchmarks import BENCHMARKS_DIR, BenchmarkConfig

        bench = BenchmarkConfig.from_config_path(BENCHMARKS_DIR / "aime24" / "config.yaml")

        assert _benchmark_domain(bench) == "math"

    def test_resolves_domain_defined_on_agent(self, tmp_path: Path) -> None:
        # `domain` can be declared on the agent (responses_api_agents.<agent>.domain) rather than on a
        # resources server, as the tau2 config does.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """tau2_agent:
  responses_api_agents:
    tau2:
      entrypoint: app.py
      domain: agent
"""
        )
        bench = MagicMock()
        bench.path = config_path

        assert _benchmark_domain(bench) == "agent"


class TestSearchBenchmarks:
    # Map each benchmark name to the `domain` its config would resolve to.
    DOMAINS = {
        "aime24": "math",
        "gpqa_diamond": "science",
    }

    def _bench(self, key: str):
        bench = MagicMock(agent_name="my_agent", num_repeats=1)
        bench.config_key = key  # let the patched _benchmark_domain find the right entry
        return bench

    def _benchmarks(self) -> dict:
        return {name: self._bench(name) for name in self.DOMAINS}

    def _run(self, query: str, benchmarks: dict, capsys) -> str:
        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config({"query": query})),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", return_value=benchmarks),
            patch("nemo_gym.cli.eval._benchmark_domain", side_effect=lambda b: self.DOMAINS[b.config_key]),
        ):
            list_benchmarks()
        return capsys.readouterr().out

    def test_query_filters_by_name(self, capsys) -> None:
        out = self._run("aime", self._benchmarks(), capsys)
        assert "aime24" in out
        assert "gpqa" not in out

    def test_query_matches_domain(self, capsys) -> None:
        # "science" only appears via gpqa's domain, not its name/agent.
        out = self._run("science", self._benchmarks(), capsys)
        assert "gpqa_diamond" in out
        assert "aime24" not in out

    def test_query_does_not_match_resource_server(self, capsys) -> None:
        # "judge" appears only in a resources server name, which is no longer searched:
        # matching is restricted to the benchmark name and domain.
        assert "No benchmarks match 'judge'" in self._run("judge", self._benchmarks(), capsys)

    def test_query_no_match_message(self, capsys) -> None:
        assert "No benchmarks match 'zzz'" in self._run("zzz", self._benchmarks(), capsys)


class TestPrepareBenchmark:
    def _make_bench_dir(self, tmp_path: Path, name: str = "fake_bench") -> tuple[Path, Path]:
        benchmarks_dir = tmp_path / "benchmarks"
        bench_dir = benchmarks_dir / name
        bench_dir.mkdir(parents=True)

        prepare_scripts_path = bench_dir / "prepare.py"
        prepare_scripts_path.write_text("")

        config_path = bench_dir / "config.yaml"
        config_path.write_text(f"""dummy_agent:
  responses_api_agents:
    simple_agent:
      datasets:
      - name: dummy_benchmark_name
        type: benchmark
        jsonl_fpath: {tmp_path / "output.jsonl"}
        prompt_config: benchmarks/dummy/prompts/default.yaml
        prepare_script: {prepare_scripts_path}
        num_repeats: 32""")

        return bench_dir, config_path

    def test_calls_prepare(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)

        mock_module = MagicMock()
        mock_module.prepare.return_value = tmp_path / "output.jsonl"

        with (
            patch(
                "nemo_gym.cli.eval.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.cli.eval.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.cli.eval.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()
            mock_module.prepare.assert_called_once()

    def test_missing_prepare_py(self, tmp_path: Path, capsys) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)
        (bench_dir / "prepare.py").unlink()

        with (
            patch(
                "nemo_gym.cli.eval.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.cli.eval.BENCHMARKS_DIR", bench_dir.parent),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_benchmark()
        assert exc_info.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "The following benchmarks are missing a valid prepare script" in out

    def test_missing_prepare_function(self, tmp_path: Path, capsys) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)

        mock_module = MagicMock()

        with (
            patch(
                "nemo_gym.cli.eval.get_global_config_dict",
                return_value=_mock_global_config(
                    {"config_paths": [str(config_path)], **safe_load(config_path.read_text())}
                ),
            ),
            patch("nemo_gym.cli.eval.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.cli.eval.importlib.import_module", return_value=mock_module),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_benchmark()
        assert exc_info.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "Expected the actual prepared dataset output fpath to match the jsonl_fpath set in the config" in out

    def test_no_benchmark_in_config_paths(self, capsys) -> None:
        with (
            patch(
                "nemo_gym.cli.eval.get_global_config_dict",
                return_value=_mock_global_config({"config_paths": ["resources_servers/foo/configs/foo.yaml"]}),
            ),
            patch("nemo_gym.cli.eval._load_benchmarks_from_config_paths", return_value={}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_benchmark()
        assert exc_info.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "No benchmark config found" in out

    def test_no_benchmark_dataset_reports_inspected_instances(self, tmp_path: Path, capsys) -> None:
        # A server instance is present but declares no `benchmark` dataset; the error should name it
        # so the user can see what was inspected.
        config = {
            "config_paths": ["benchmarks/dummy/config.yaml"],
            "dummy_agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "datasets": [{"name": "not_a_benchmark", "type": "train", "jsonl_fpath": str(tmp_path)}]
                    }
                }
            },
        }
        with patch("nemo_gym.cli.eval.get_global_config_dict", return_value=_mock_global_config(config)):
            with pytest.raises(SystemExit) as exc_info:
                prepare_benchmark()
        assert exc_info.value.code == 1
        out = " ".join(capsys.readouterr().out.split())
        assert "Inspected server instances ['dummy_agent']" in out

    def test_caching_sanity(self, tmp_path: Path) -> None:
        bench_dir, config_path = self._make_bench_dir(tmp_path)
        (tmp_path / "output.jsonl").write_text("blah blah text for file")

        mock_module = MagicMock()
        mock_module.prepare.return_value = tmp_path / "output.jsonl"

        with (
            patch(
                "nemo_gym.cli.eval.get_global_config_dict",
                return_value=_mock_global_config(
                    {
                        "use_cached_prepared_benchmarks": True,
                        "config_paths": [str(config_path)],
                        **safe_load(config_path.read_text()),
                    }
                ),
            ),
            patch("nemo_gym.cli.eval.BENCHMARKS_DIR", bench_dir.parent),
            patch("nemo_gym.cli.eval.importlib.import_module", return_value=mock_module),
        ):
            prepare_benchmark()

        assert mock_module.prepare.call_count == 0
