# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import asyncio
from dataclasses import dataclass, is_dataclass
import glob
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import threading
import time
from collections import defaultdict

import httpx


def nested_dataclass(*args, **kwargs):
    def wrapper(check_class):
        try:
            from omegaconf.dictconfig import DictConfig

            dict_types = (dict, DictConfig)
        except ImportError:
            dict_types = (dict,)

        check_class = dataclass(check_class, **kwargs)
        orig_init = check_class.__init__

        def __init__(self, *, _init_nested=False, **init_kwargs):
            if _init_nested:
                for name, value in init_kwargs.items():
                    field_type = check_class.__annotations__.get(name)
                    if is_dataclass(field_type) and isinstance(value, dict_types):
                        init_kwargs[name] = field_type(**value, _init_nested=_init_nested)
            orig_init(self, **init_kwargs)

        check_class.__init__ = __init__
        return check_class

    return wrapper(args[0]) if args else wrapper


@nested_dataclass(kw_only=True)
class BaseEvaluatorConfig:
    input_file: str | None = None
    data_dir: str | None = None
    split: str = "test"


class BaseEvaluator:
    def __init__(self, config: dict, num_parallel_requests: int = 10):
        self.config = config
        self.num_parallel_requests = num_parallel_requests

    async def eval_single(self, data_point: dict):
        raise NotImplementedError


def unroll_files(input_files, parent_dir: str | None = None):
    if len(input_files) == 0:
        raise ValueError("No files found with the given pattern.")

    total_files = 0
    for file_pattern in input_files:
        if parent_dir is not None:
            file_pattern = os.path.join(parent_dir, file_pattern)
        for file in sorted(glob.glob(file_pattern, recursive=True)):
            total_files += 1
            yield file

    if total_files == 0:
        raise ValueError("No files found with the given pattern.")


def jdump(obj, f, mode="w", indent=None, default=str):
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"Expected a single or list of dictionaries, but got {type(obj)}.")

    parent = os.path.dirname(f)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(f, mode=mode, encoding="utf-8") as handle:
        for line in obj:
            json.dump(line, handle, indent=indent, default=default)
            handle.write("\n")


class LocalSandbox:
    def __init__(
        self,
        host: str = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: str = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
    ):
        self.host = host
        self.port = port
        self.http_session = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=2048, max_connections=2048),
        )

    async def close(self):
        await self.http_session.aclose()

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = "ipython",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id=None,
        traceback_verbosity="plain",
    ):
        if session_id is not None:
            raise RuntimeError("Stateful execution is not supported by this sandbox client.")

        request = {
            "generated_code": generated_code,
            "std_input": std_input,
            "timeout": timeout,
            "language": language,
            "max_output_characters": max_output_characters,
            "traceback_verbosity": traceback_verbosity.capitalize(),
        }
        try:
            response = await self.http_session.post(
                url=f"http://{self.host}:{self.port}/execute",
                content=json.dumps(request),
                timeout=timeout + 5.0,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 502:
                raise httpx.TimeoutException("502 error")
            try:
                output = response.json()
            except json.JSONDecodeError:
                output = {"process_status": "error", "stdout": "", "stderr": "Unknown error"}
        except (httpx.TimeoutException, httpx.TransportError):
            output = {"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"}
        return output, None


@nested_dataclass(kw_only=True)
class CCCEvaluatorConfig(BaseEvaluatorConfig):
    test_file: str = "test_metadata.jsonl"
    num_workers: int = 32
    test_batch_size: int = 32
    time_scale: float = 2.0
    overwrite: bool = False
    shared_dir: str = "/tmp"


_precompile_loop_tls = threading.local()
_test_loop_tls = threading.local()
_test_sandbox_tls = threading.local()


def _exec_sync(tls: threading.local, sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    loop = getattr(tls, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        tls.loop = loop
    return loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def _get_thread_test_sandbox() -> LocalSandbox:
    sandbox = getattr(_test_sandbox_tls, "sandbox", None)
    if sandbox is None:
        sandbox = LocalSandbox()
        _test_sandbox_tls.sandbox = sandbox
    return sandbox


def wait_for_sandbox(sandbox, timeout: int = 240, poll: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _exec_sync(_precompile_loop_tls, sandbox, "echo hello world", language="shell", timeout=10)
            if resp.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        time.sleep(poll)
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def _safe_fs_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _looks_like_problem_metadata(problem_metadata: dict) -> bool:
    return isinstance(problem_metadata, dict) and (
        "grader_files" in problem_metadata
        and "compile" in problem_metadata
        and "run" in problem_metadata
        and ("all_tests" in problem_metadata or "tests" in problem_metadata)
    )


def _looks_like_legacy_problem_metadata(problem_metadata: dict) -> bool:
    return (
        isinstance(problem_metadata, dict)
        and bool(problem_metadata)
        and all(
            isinstance(subtask_metadata, dict)
            and "grader_files" in subtask_metadata
            and "compile" in subtask_metadata
            and "run" in subtask_metadata
            and "tests" in subtask_metadata
            for subtask_metadata in problem_metadata.values()
        )
    )


def _normalize_problem_metadata(problem_id: str, problem_metadata: dict) -> dict:
    if "subtasks" in problem_metadata and "all_tests" in problem_metadata:
        return problem_metadata

    normalized_subtasks = {}
    normalized_tests = {}
    compile_code = ""
    run_code = ""
    grader_files = []
    for subtask_name, subtask_metadata in problem_metadata.items():
        if not isinstance(subtask_metadata, dict):
            continue

        compile_code = compile_code or subtask_metadata.get("compile", "")
        run_code = run_code or subtask_metadata.get("run", "")
        grader_files = grader_files or subtask_metadata.get("grader_files", [])

        test_names = []
        sample_test_names = []
        secret_test_names = []
        for test_name, test_data in (subtask_metadata.get("tests") or {}).items():
            test_names.append(test_name)
            group = "sample" if test_name.startswith("sample") else "secret"
            if group == "sample":
                sample_test_names.append(test_name)
            else:
                secret_test_names.append(test_name)
            normalized_tests.setdefault(
                test_name,
                {
                    "input": test_data["input"],
                    "output": test_data["output"],
                    "group": group,
                },
            )

        normalized_subtasks[subtask_name] = {
            "aggregation": "min",
            "score": float(subtask_metadata.get("score") or subtask_metadata.get("subtask_score") or 0.0),
            "score_precision": int(subtask_metadata.get("score_precision", 0)),
            "test_names": test_names,
            "sample_test_names": sample_test_names,
            "secret_test_names": secret_test_names,
        }

    return {
        "name": problem_metadata.get("name", problem_id),
        "compile": compile_code,
        "run": run_code,
        "grader_files": grader_files,
        "subtasks": normalized_subtasks,
        "all_tests": normalized_tests,
    }


def _iter_metadata_entries(raw_metadata):
    if isinstance(raw_metadata, list):
        for entry in raw_metadata:
            yield entry
        return

    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Unsupported metadata payload type: {type(raw_metadata).__name__}")

    if "metadata" in raw_metadata or "problems" in raw_metadata or "competition_id" in raw_metadata or "competition" in raw_metadata:
        yield raw_metadata
        return

    if raw_metadata and all(
        _looks_like_problem_metadata(v) or _looks_like_legacy_problem_metadata(v) for v in raw_metadata.values()
    ):
        yield {"metadata": raw_metadata}
        return

    for competition_id, problems in raw_metadata.items():
        if isinstance(problems, dict):
            yield {"competition_id": competition_id, "metadata": problems}


def _load_metadata_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw_metadata = [json.loads(line) for line in f if line.strip()]

    metadata_by_competition = {}
    problem_index = defaultdict(dict)

    for entry in _iter_metadata_entries(raw_metadata):
        competition_id = str(entry.get("competition_id") or entry.get("competition") or "")
        problems = entry.get("metadata") or entry.get("problems") or {}
        normalized_problems = {}
        for problem_id, problem_metadata in problems.items():
            normalized = _normalize_problem_metadata(problem_id, problem_metadata)
            normalized_problems[problem_id] = normalized
            problem_index[problem_id][competition_id] = normalized
        metadata_by_competition.setdefault(competition_id, {}).update(normalized_problems)

    return metadata_by_competition, dict(problem_index)


def _precompile_problem(
    problem_key: str,
    problem_id: str,
    grader_files,
    compile_code: str,
    run_code: str,
    sandbox: LocalSandbox,
    shared_dir: str,
) -> str:
    if getattr(sandbox, "_owner_tid", None) != threading.get_ident():
        sandbox = LocalSandbox()
        wait_for_sandbox(sandbox)
        sandbox._owner_tid = threading.get_ident()

    pre_dir = f"{shared_dir}/ccc_pre_{_safe_fs_name(problem_key)}_{os.getpid()}"
    os.makedirs(os.path.join(pre_dir, "graders"), exist_ok=True)

    for filepath, content in grader_files:
        target_path = os.path.join(pre_dir, filepath)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

    for script_name, script_content in (("compile.sh", compile_code), ("run.sh", run_code)):
        script_path = os.path.join(pre_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

    _exec_sync(_precompile_loop_tls, sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)
    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    unique_dir = f"{task_args['shared_dir']}/ccc_run_{worker_id}_{os.getpid()}_{time.time_ns()}"
    try:
        precompiled_dir = task_args.get("precompiled_dir")
        os.makedirs(unique_dir, exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "graders"), exist_ok=True)
        os.makedirs(os.path.join(unique_dir, "tmp"), exist_ok=True)
        if precompiled_dir and os.path.isdir(precompiled_dir):
            shutil.copytree(precompiled_dir, unique_dir, dirs_exist_ok=True)
        if task_args.get("task_type") == "SIMULATION":
            with open(os.path.join(unique_dir, "solution.odo"), "w", encoding="utf-8") as f:
                f.write(task_args["generated_code"])
        else:
            with open(os.path.join(unique_dir, "graders", f"{task_args['problem_id']}.cpp"), "w", encoding="utf-8") as f:
                f.write(task_args["generated_code"])
        with open(os.path.join(unique_dir, "input.txt"), "w", encoding="latin1") as f:
            f.write(task_args["test_input"])
        with open(os.path.join(unique_dir, "correct_output.txt"), "w", encoding="latin1") as f:
            f.write(task_args["test_output"])

        sandbox = _get_thread_test_sandbox()
        compile_result = _exec_sync(_test_loop_tls, sandbox, f"cd {unique_dir} && ./compile.sh", language="shell", timeout=120)
        result = {
            "compile_success": not compile_result.get("stderr"),
            "compile_stdout": compile_result.get("stdout", ""),
            "compile_stderr": compile_result.get("stderr", ""),
            "run_stdout": "",
            "run_stderr": "",
            "error": "",
            "score": 0.0,
        }
        if not result["compile_success"]:
            return result

        run_timeout = max(1, int(120 * float(task_args.get("time_scale", 1.0))))
        run_start = time.monotonic()
        run_result = _exec_sync(
            _test_loop_tls,
            sandbox,
            f"cd {unique_dir} && export TMPDIR={unique_dir}/tmp && TIME_LIMIT_SCALE={task_args.get('time_scale', 1.0)} ./run.sh",
            language="shell",
            timeout=run_timeout,
        )
        result["run_time_s"] = time.monotonic() - run_start
        result["run_stdout"] = run_result.get("stdout", "")
        result["run_stderr"] = run_result.get("stderr", "")
        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0
        return result
    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}
    finally:
        try:
            shutil.rmtree(unique_dir, ignore_errors=True)
        except Exception:
            pass


def extract_final_code_block(text: str, langs: tuple[str, ...]) -> str:
    pattern = r"```(?:" + "|".join(re.escape(l) for l in langs) + r")\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1] if matches else (text or "")


def extract_task_config(problem_metadata: dict) -> dict:
    for relpath, content in problem_metadata.get("grader_files", []):
        if relpath == "graders/grader_config.json":
            try:
                return json.loads(content)
            except Exception:
                return {}
    return {}


def add_includes(code: str, problem_header_include: str | None = None, problem_id: str | None = None) -> str:
    if not code:
        return code
    code_header = "#include <bits/stdc++.h>\n"
    if problem_header_include:
        header_include = f'#include "{problem_header_include}"'
        if header_include not in code:
            code_header += header_include + "\n"
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + (("\n" + dummy) if dummy else ("\n" if not code.endswith("\n") else ""))


class CCCEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests: int = 20):
        super().__init__(config, num_parallel_requests)
        self.eval_cfg = CCCEvaluatorConfig(_init_nested=True, **config)
        self.sandbox = None
        self.metadata = None
        self.metadata_by_competition = None
        self.problem_index = None
        self.precompiled_cache = {}
        self.pool = None
        self._init_lock = asyncio.Lock()

    async def _initialize_runtime(self):
        if self.sandbox is not None:
            return

        async with self._init_lock:
            if self.sandbox is not None:
                return

            def _setup():
                sbox = LocalSandbox()
                wait_for_sandbox(sbox)
                sbox._owner_tid = threading.get_ident()
                if not os.path.exists(self.eval_cfg.test_file):
                    raise FileNotFoundError(f"Metadata file {self.eval_cfg.test_file} does not exist.")
                metadata_by_competition_local, problem_index_local = _load_metadata_file(self.eval_cfg.test_file)
                pool_local = ThreadPoolExecutor(max_workers=self.eval_cfg.test_batch_size)
                return sbox, metadata_by_competition_local, problem_index_local, pool_local

            self.sandbox, self.metadata_by_competition, self.problem_index, self.pool = await asyncio.to_thread(_setup)
        # Keep a flat fast-path for legacy callers when problem ids are globally unique.
        self.metadata = {
            problem_id: next(iter(problem_metadata_by_comp.values()))
            for problem_id, problem_metadata_by_comp in self.problem_index.items()
            if len(problem_metadata_by_comp) == 1
        }

    def _get_competition_id(self, entry: dict) -> str | None:
        competition_id = entry.get("competition_id") or entry.get("competition")
        return str(competition_id) if competition_id not in (None, "") else None

    def _cache_key(self, problem_id: str, competition_id: str | None) -> str:
        return f"{competition_id}::{problem_id}" if competition_id else problem_id

    def get_problem_metadata(self, problem_id: str, competition_id: str | None = None) -> dict:
        if competition_id:
            competition_problems = (self.metadata_by_competition or {}).get(competition_id)
            if competition_problems is None:
                available = sorted(k for k in (self.metadata_by_competition or {}).keys() if k)
                raise ValueError(
                    f"Competition '{competition_id}' not found for problem '{problem_id}'. Available competitions: {available}"
                )
            if problem_id not in competition_problems:
                available_problems = sorted(competition_problems.keys())
                raise ValueError(
                    f"Problem '{problem_id}' not found in competition '{competition_id}'. Available problems: {available_problems}"
                )
            return competition_problems[problem_id]

        matches = (self.problem_index or {}).get(problem_id, {})
        if len(matches) == 1:
            return next(iter(matches.values()))
        if not matches:
            raise ValueError(f"Problem '{problem_id}' not found in metadata.")

        available_competitions = sorted(k for k in matches.keys() if k)
        raise ValueError(
            f"Problem '{problem_id}' exists in multiple competitions; please provide competition_id. "
            f"Available competitions: {available_competitions}"
        )

    def _get_precompiled_dir(self, cache_key: str, problem_id: str, problem_metadata: dict):
        if cache_key in self.precompiled_cache:
            cached = self.precompiled_cache[cache_key]
            return cached["grader"] if isinstance(cached, dict) else cached

        grader_dir = _precompile_problem(
            cache_key,
            problem_id,
            problem_metadata["grader_files"],
            problem_metadata["compile"],
            problem_metadata["run"],
            self.sandbox,
            self.eval_cfg.shared_dir,
        )
        self.precompiled_cache[cache_key] = {"grader": grader_dir}
        return grader_dir

    def _build_test_task(self, problem_id: str, pre_dir: str, completion: str, test_data: dict, task_type: str = "Batch"):
        return {
            "generated_code": completion,
            "task_type": task_type,
            "problem_id": problem_id,
            "precompiled_dir": pre_dir,
            "test_input": test_data["input"],
            "test_output": test_data["output"],
            "time_scale": self.eval_cfg.time_scale,
            "shared_dir": self.eval_cfg.shared_dir,
        }

    def _aggregate_subtask_score(self, subtask_meta: dict, outputs: list[dict], failed: bool = False) -> float:
        aggregation = subtask_meta["aggregation"]
        if aggregation == "min":
            if failed:
                return 0.0
            scores = [float(out.get("score", 0.0)) for out in outputs]
            precision = max(0, int(subtask_meta.get("score_precision", 0)))
            return round(
                (min(scores) if scores else 0.0) * float(subtask_meta["score"]),
                precision,
            )
        if aggregation == "sum_tests":
            return float(sum(1 for out in outputs if float(out.get("score", 0.0)) > 0.0))
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    async def _evaluate_entry(self, entry: dict) -> dict:
        await self._initialize_runtime()

        problem_id = entry.get("problem_id") or entry.get("ioi_id")
        if not problem_id:
            raise ValueError("Missing 'problem_id' field in entry")

        competition_id = self._get_competition_id(entry)
        problem_metadata = self.get_problem_metadata(problem_id, competition_id)
        task_config = extract_task_config(problem_metadata)
        task_type = str(task_config.get("task_type", "Batch"))
        if task_type == "SIMULATION":
            completion = extract_final_code_block((entry.get("generation") or ""), ("txt", "text", "plain"))
        elif task_type == "MULTIFILE":
            completion = extract_final_code_block((entry.get("generation") or ""), ("cpp",))
        else:
            completion = add_includes(
                extract_final_code_block((entry.get("generation") or ""), ("cpp",)),
                problem_metadata.get("problem_header_include"),
                problem_id,
            )
        cache_key = self._cache_key(problem_id, competition_id)
        pre_dir = await asyncio.to_thread(self._get_precompiled_dir, cache_key, problem_id, problem_metadata)

        subtask_state = {
            subtask_name: {
                "aggregation": subtask_meta["aggregation"],
                "outputs": [],
                "failed": False,
            }
            for subtask_name, subtask_meta in problem_metadata["subtasks"].items()
        }
        test_to_subtasks = {}
        for subtask_name, subtask_meta in problem_metadata["subtasks"].items():
            for test_name in subtask_meta["test_names"]:
                test_to_subtasks.setdefault(test_name, []).append(subtask_name)

        all_test_items = list(problem_metadata["all_tests"].items())
        batch_size = self.eval_cfg.test_batch_size
        all_run_times: list[float] = []
        for i in range(0, len(all_test_items), batch_size):
            candidate_batch = all_test_items[i : i + batch_size]
            batch = []
            tasks = []
            for test_name, test_data in candidate_batch:
                subtasks = test_to_subtasks.get(test_name, [])
                should_run = False
                for subtask_name in subtasks:
                    state = subtask_state[subtask_name]
                    if state["aggregation"] == "sum_tests" or not state["failed"]:
                        should_run = True
                        break
                if not should_run:
                    continue
                batch.append((test_name, test_data))
                tasks.append(self._build_test_task(problem_id, pre_dir, completion, test_data, task_type=task_type))
            if not batch:
                continue
            loop = asyncio.get_running_loop()
            futures = [loop.run_in_executor(self.pool, run_test_case, task, idx) for idx, task in enumerate(tasks)]
            results = await asyncio.gather(*futures)
            for (test_name, _), result in zip(batch, results):
                if "run_time_s" in result:
                    all_run_times.append(result["run_time_s"])
                result["test_name"] = test_name
                test_group = problem_metadata["all_tests"][test_name].get("group")
                if test_group is not None:
                    result["test_group"] = test_group
                for subtask_name in test_to_subtasks.get(test_name, []):
                    state = subtask_state[subtask_name]
                    if state["aggregation"] == "min" and state["failed"]:
                        continue
                    state["outputs"].append(dict(result))
                    if state["aggregation"] == "min" and float(result.get("score", 0.0)) == 0.0:
                        state["failed"] = True

        test_case_results = {}
        for subtask_name, subtask_meta in problem_metadata["subtasks"].items():
            state = subtask_state[subtask_name]
            test_case_results[subtask_name] = {
                "score": self._aggregate_subtask_score(subtask_meta, state["outputs"], failed=state["failed"]),
                "outputs": state["outputs"],
            }


        num_tests_run = len(all_run_times)
        return {
            "name": entry["name"],
            "subtask": entry["subtask"],
            "test_case_results": test_case_results,
            "num_tests_run": num_tests_run,
            "total_test_execution_time_s": sum(all_run_times),
            "mean_test_execution_time_s": sum(all_run_times) / num_tests_run if num_tests_run else 0.0,
        }

    async def eval_full(self, input_files):  # type: ignore[override]
        await self._initialize_runtime()

        for jsonl_file in unroll_files(input_files):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                all_samples = [json.loads(line) for line in f]

            # Precompile each unique problem once before row-level concurrency starts.
            unique_problem_keys = []
            seen_problem_keys = set()
            for sample in all_samples:
                competition_id = self._get_competition_id(sample)
                problem_id = sample.get("problem_id") or sample.get("ioi_id")
                if not problem_id:
                    raise ValueError("Missing 'problem_id' field in eval_full sample")
                cache_key = self._cache_key(problem_id, competition_id)
                if cache_key not in seen_problem_keys:
                    seen_problem_keys.add(cache_key)
                    unique_problem_keys.append((cache_key, problem_id, competition_id))

            for cache_key, problem_id, competition_id in unique_problem_keys:
                problem_metadata = self.get_problem_metadata(problem_id, competition_id)
                await asyncio.to_thread(self._get_precompiled_dir, cache_key, problem_id, problem_metadata)

            tasks = [self._evaluate_entry(sample) for sample in all_samples]
            outputs = await asyncio.gather(*tasks)
            for sample, output in zip(all_samples, outputs):
                sample["test_case_results"] = output["test_case_results"]
            jdump(all_samples, jsonl_file, mode="wt")

        if self.pool is not None:
            self.pool.shutdown(wait=True)

    async def eval_single(self, data_point: dict):
        return await self._evaluate_entry(data_point)