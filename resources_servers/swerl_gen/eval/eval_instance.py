# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import base64
import json
import logging
import os
import pickle
import re
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TypedDict


logging.basicConfig(level=logging.INFO)


repo_list = ["sympy", "django", "pytest", "default"]

APPLY_PATCH_FAIL = "error: patch"
RESET_FAILED = "Reset Failed"
TESTS_ERROR = "Tests Errored"
TESTS_TIMEOUT = "Tests Timed Out"


class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


# Taken from SWEbench
# Constants - Task Instance Class
class SWEbenchInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"


def looks_like_path(path):
    return isinstance(path, str) and (
        os.path.isabs(path) or os.path.sep in path or os.path.altsep and os.path.altsep in path
    )


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)


def read_problems_list(evalset_file) -> Dict[str, Dict]:
    problems = {}
    for task in stream_jsonl(evalset_file):
        if task["instance_id"] not in problems:
            problems[task["instance_id"]] = [task]
        else:
            problems[task["instance_id"]].append(task)
    return problems


def parse_eval_output(line):
    success_pattern = re.compile(r"=+ (\d+) passed.*$")
    failure_pattern = re.compile(r"=+ (\d+) failed, (\d+) passed.*$")

    failure_match = failure_pattern.match(line)
    if failure_match:
        failed_count, passed_count = map(int, failure_match.groups())
        return failed_count, passed_count

    # Check for success pattern
    success_match = success_pattern.match(line)
    if success_match:
        passed_count = int(success_match.group(1))
        return 0, passed_count

    return -1, -1


def parse_xml_eval_output(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for testsuite in root.findall("testsuite"):
        out = {
            "error_count": int(testsuite.get("errors")),
            "failed_count": int(testsuite.get("failures")),
            "skipped_count": int(testsuite.get("skipped")),
            "passed_count": int(testsuite.get("tests"))
            - (int(testsuite.get("errors")) + int(testsuite.get("failures")) + int(testsuite.get("skipped"))),
            "total_count": int(testsuite.get("tests")),
        }
        if out:
            return out
    out = None


def parse_pytest_eval_output(log):
    stat = {
        "error_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "passed_count": 0,
        "total_count": 0,
    }
    for test in ["passed", "skipped", "failed"]:
        pattern = re.compile(rf"(\d+) \b{test}\b")
        match = pattern.search(log)
        if match:
            stat[f"{test}_count"] = int(match.group(1))
    stat["total_count"] = sum([count for key, count in stat.items() if key != "total_count"])

    return stat


def get_testname(name):
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    has_option = option_pattern.search(name)
    if has_option:
        main, option = has_option.groups()
        if (
            option.startswith("/") and not option.startswith("//") and "*" not in option and "-/" not in option
        ):  ### updated condition for pallets__flask-5014
            option = "/" + option.split("/")[-1]
        test_name = f"{main}[{option}]"
    else:
        test_name = name
    return test_name


def detailed_parse_pytest_eval_output_v2(log):
    stat = {
        "error_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "passed_count": 0,
        "total_count": 0,
    }
    for test in ["passed", "skipped", "failed"]:
        pattern = re.compile(rf"(\d+) \b{test}\b")
        match = pattern.search(log)
        if match:
            stat[f"{test}_count"] = int(match.group(1))
    stat["total_count"] = sum([count for key, count in stat.items() if key != "total_count"])

    #### taken from SWEbench
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])

    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        line = re.sub(r"\s*\[\s*\d+\s*%\s*]$", "", line)  ### remove [ d%] at the end of the line
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if "tests/test_main.py::test_model_post_init_supertype_private_attr" in line:
            print(line)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[get_testname(test_case[1])] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[get_testname(test_case[0])] = test_case[1]

    stat.update({"test_status_map": test_status_map})

    return stat


def detailed_parse_pytest_eval_output(log):
    stat = {
        "error_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "passed_count": 0,
        "total_count": 0,
    }
    for test in ["passed", "skipped", "failed"]:
        pattern = re.compile(rf"(\d+) \b{test}\b")
        match = pattern.search(log)
        if match:
            stat[f"{test}_count"] = int(match.group(1))
    stat["total_count"] = sum([count for key, count in stat.items() if key != "total_count"])

    #### taken from SWEbench
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[0]] = test_case[1]

    stat.update({"test_status_map": test_status_map})

    return stat


def detailed_parse_django_eval_output(log) -> dict[str, str]:
    """
    (taken from SWE-bench repo)
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    stat = {
        "error_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "passed_count": 0,
        "total_count": 0,
    }

    prev_test = None
    for line in lines:
        line = line.strip()
        line = line.replace("…", "...")
        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK", "... OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                stat["passed_count"] += 1
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
            stat["skipped_count"] += 1
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
            stat["failed_count"] += 1
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
            stat["failed_count"] += 1
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
            stat["error_count"] += 1
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value
            stat["error_count"] += 1

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            stat["passed_count"] += 1
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
            stat["passed_count"] += 1

    stat["total_count"] = sum([count for key, count in stat.items() if key != "total_count"])
    stat.update({"test_status_map": test_status_map})
    return stat


def detailed_parse_sympy_eval_output(log: str) -> dict[str, str]:
    """
    (taken from SWE-bench repo with small modifications)
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    stat = {
        "error_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "passed_count": 0,
        "total_count": 0,
    }
    test_status_map = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split("\n"):
        line = line.replace("[OK]", "")
        line = line.replace("[FAIL]", "")
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
                stat["error_count"] += 1
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
                stat["failed_count"] += 1
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
                stat["passed_count"] += 1
        elif TestStatus.PASSED.value in line:
            parts = line.split()
            if len(parts) > 1:
                test = parts[0]
                if parts[0] == TestStatus.PASSED.value:
                    test = parts[1]
                test_status_map[test] = TestStatus.PASSED.value
                stat["passed_count"] += 1

    stat["total_count"] = sum([count for key, count in stat.items() if key != "total_count"])
    stat.update({"test_status_map": test_status_map})
    return stat


PARSER_FUNCS = {
    repo_name: (
        detailed_parse_sympy_eval_output
        if repo_name == "sympy"
        else detailed_parse_django_eval_output
        if repo_name == "django"
        else detailed_parse_pytest_eval_output
        if repo_name == "pytest"
        else detailed_parse_pytest_eval_output_v2
    )
    for repo_name in repo_list
}


def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] == TestStatus.PASSED.value


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or any(sm[case] == status for status in [TestStatus.FAILED.value, TestStatus.ERROR.value])


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value, p2p, f2p
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value, p2p, f2p
    else:
        return ResolvedStatus.NO.value, p2p, f2p


def get_eval_tests_report(
    eval_sm: dict[str, str],
    gold_results: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    """
    (Taken from SWEbench)
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """
    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        if test_passed(test_case, eval_sm):
            # Assume silent success for now (test case not in eval_sm)
            f2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            f2p_failure.append(test_case)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        test_case = test_case.replace(".*\\\\(1", "")  ### for 14 cases of scikit-learn__scikit-learn-25570
        if test_passed(test_case, eval_sm):
            p2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            p2p_failure.append(test_case)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    return results


def analyze_eval_tests(instance, test_map):
    def _get_present(instance, *keys):
        for k in keys:
            if k in instance:
                v = instance[k]
                return json.loads(v) if isinstance(v, str) else v
        return []

    eval_ref = {
        "instance_id": instance["instance_id"],
        FAIL_TO_PASS: _get_present(instance, "FAIL_TO_PASS", "fail_to_pass_select", "fail_to_pass"),
        PASS_TO_PASS: _get_present(instance, "PASS_TO_PASS", "pass_to_pass_select", "pass_to_pass"),
    }

    report = get_eval_tests_report(test_map, eval_ref)

    return get_resolution_status(report)[0]


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def update_files(edit_file_path, base_dir, patch=None):
    if patch:
        content = (
            "#!/bin/bash\n"  # Add shebang line
            + f"cd {base_dir}\n"  # Add newline for better readability
            + "\ngit apply -v - <<'EOF_114329324912'\n"
            + (f"{patch}")
            + "\nEOF_114329324912\n\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_script:
            temp_script.write(content)
            temp_script.flush()
            os.fsync(temp_script.fileno())  # Ensure all data is written to disk
            os.chmod(temp_script.name, os.stat(temp_script.name).st_mode | stat.S_IEXEC)

        try:
            result = subprocess.run([temp_script.name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info("Update patch script executed successfully.")
            out = result.stdout.replace("…", "...")
            err = result.stderr.replace("…", "...")
            content = out + err
        except subprocess.CalledProcessError as e:
            logging.error("Update patch script execution failed.")
            out = e.stdout.replace("…", "...")
            err = e.stderr.replace("…", "...")
            content = out + err
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_script.name)
            except OSError:
                pass

        return content
    else:
        if not edit_file_path.startswith("/"):
            edit_file_path = f"{get_root_path()}/logs/runs/pickles/{edit_file_path}"  ## the default log path
        logging.info(f"Edit file path: {edit_file_path}")
        assert os.path.exists(edit_file_path)
        edited_python_files = pickle.load(open(edit_file_path, "rb"))
        for file in edited_python_files["edited_files"]:
            filename = os.path.join("/testbed", "/".join(file.split("/")[1:]))

            if os.path.exists(filename):
                logging.info(f"Update file {file} at {filename}")
            else:
                logging.info(f"Create a new file {file} at {filename}")

            with open(filename, "w") as f:
                content = "\n".join(edited_python_files["python_files"][file]["text"])
                f.write(content)

        return "Successfully updated files."


def get_bash_file_path(instance_id, base_dir, setup=True, regression=False):
    os.makedirs(base_dir, exist_ok=True)
    script_path = (
        os.path.join(base_dir, f"{instance_id}_regression.sh")
        if regression
        else os.path.join(base_dir, f"{instance_id}.sh")
        if setup
        else os.path.join(base_dir, f"{instance_id}_test.sh")
    )
    if os.path.exists(script_path):
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    return script_path


def make_reproduction_commands(reproduce_patch):
    env_name = "testbed"
    repo_directory = f"/{env_name}"

    # Some test_cmd seem to be slightly different. Double check.
    HEREDOC_DELIMITER = "EOF_114329324912"
    apply_reproduce_test_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{reproduce_patch}\n{HEREDOC_DELIMITER}"

    eval_commands = [
        "source /opt/miniconda3/bin/activate && ",
        f"conda activate {env_name} && ",
        f"cd {repo_directory} && ",
    ]

    eval_commands += [apply_reproduce_test_command]

    return "\n".join(eval_commands)


def format_output_stream(result):
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    out = result.stdout.replace("…", "...")
    out = ansi_escape.sub("", out)
    err = result.stderr.replace("…", "...")
    err = ansi_escape.sub("", err)
    return out + err


def run_bash_script(
    script_path: Optional[str] = None,
    script_content: Optional[str] = None,
):
    """
    Run a bash script either from a filesystem path or from in-memory content.

    Using in-memory content avoids dependence on local script files and adds
    only the overhead of spawning a single bash process.
    """
    if script_content is not None:
        return subprocess.run(
            ["bash", "-c", script_content],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
    if not script_path:
        raise ValueError("Either script_path or script_content must be provided.")
    return subprocess.run(
        [script_path],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )


def run_evaluation_on_instance(
    instance,
    instance_stats,
    script_dir: Optional[str] = None,
    edit_file=True,
    setup_script_content: Optional[str] = None,
    test_script_content: Optional[str] = None,
    output_dir: Optional[str] = Path("/root"),
):
    instance_id = instance["instance_id"]
    repo_name = instance["repo"].split("/")[-1]
    # If script contents are provided, we don't need local script files.
    script_setup_path = None
    script_test_path = None
    if setup_script_content is None or test_script_content is None:
        if not script_dir:
            raise ValueError(
                "script_dir must be provided if setup_script_content or test_script_content is not provided"
            )
        script_setup_path = get_bash_file_path(instance_id, script_dir, setup=True, regression=False)
        script_test_path = get_bash_file_path(instance_id, script_dir, setup=False, regression=False)

    instance_comes_with_parser = False
    base_dir = instance.get("base_dir", "/testbed")
    if "run_script.sh" in instance and "parsing_script.py" in instance:
        instance_comes_with_parser = True
        run_script = instance["run_script.sh"]
        parsing_script = instance["parsing_script.py"]
        run_script_path = output_dir / "run_script.sh"
        parsing_script_path = output_dir / "parsing_script.py"
        with open(run_script_path, "w") as f:
            f.write(run_script)
        with open(parsing_script_path, "w") as f:
            f.write(parsing_script)

    # Define a regex pattern to match ANSI escape codes to remove color from output
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    update_status = None
    try:
        result = run_bash_script(script_path=script_setup_path, script_content=setup_script_content)
        logging.info("Setup script executed successfully.")
        out = result.stdout.replace("…", "...")
        out = ansi_escape.sub("", out)
        err = result.stderr.replace("…", "...")
        err = ansi_escape.sub("", err)
        content = out + err
        logging.info(content)

        # apply edits
        if edit_file:
            patch = None
            edit_file_path = None
            if "model_patch" in instance_stats[instance_id]:
                patch = instance_stats[instance_id]["model_patch"]
            else:
                raise ValueError("model_patch must be provided in instance_stats")

            update_status = update_files(edit_file_path, base_dir, patch)
            content += update_status
        else:
            update_status = ""
        # run test script
        result = run_bash_script(script_path=script_test_path, script_content=test_script_content)
        logging.info("Test script executed successfully.")
        out = result.stdout.replace("…", "...")
        out = ansi_escape.sub("", out)
        err = result.stderr.replace("…", "...")
        err = ansi_escape.sub("", err)
        content += out + err
    except subprocess.CalledProcessError as e:
        logging.error("Script execution failed.")
        out = e.stdout.replace("…", "...")
        err = e.stderr.replace("…", "...")
        content = out + err
    logging.info(out + err)
    if (
        any(
            [
                x in content
                for x in [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                    "Failed to reset task environment",
                    "Could not fix",
                ]
            ]
        )
        or "applied patch" not in content.lower()
    ):
        # Eval patch was not applied successfully
        instance_stats[instance_id]["resolution"] = ResolvedStatus.NO.value
    else:
        if instance_comes_with_parser:
            eval_results_file = output_dir / "output.json"
            with open(eval_results_file, "r") as f:
                eval_results_content = json.load(f)
                test_results = eval_results_content.get("tests", {})
            instance_eval_results = {"test_status_map": {t.get("name", ""): t.get("status", "") for t in test_results}}
        else:
            if repo_name not in repo_list:
                repo_name = "default"
            instance_eval_results = PARSER_FUNCS[repo_name](content)
        instance_stats[instance_id]["resolution"] = analyze_eval_tests(
            instance, instance_eval_results["test_status_map"]
        )

        del instance_eval_results["test_status_map"]

    return instance_stats


def run_reproduction_on_instance_single(test_patch_command, index):
    """
    Run a script to reproduce a bug.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, f"temp_{index}.out")
            # redirecting python script output to temp
            test_patch_command = (
                test_patch_command
                + f"\npython reproduce_bug_{index}.py > {temp_file} 2>&1"
                + f"\necho -e '\nreproduction test status:'$? >> {temp_file}"
            )
            patch_result = subprocess.run(
                ["bash", "-c", test_patch_command],
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            patch_console_output = patch_result.stdout + patch_result.stderr
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    result = f.read()
            else:
                result = "Setup issue\nResult file not exist."
        return {
            "patch_output": patch_console_output,
            "test_output": result,
            "reproduction_test_index": index,
            "test_patch": test_patch_command,
        }
    except subprocess.CalledProcessError as e:
        out = e.stdout.replace("…", "...")
        err = e.stderr.replace("…", "...")
        content = "Setup issues\n" + out + err
        return {
            "patch_output": patch_console_output,
            "test_output": content,
            "reproduction_test_index": index,
            "test_patch": test_patch_command,
        }


def run_reproduction_on_instance(
    instance,
    instance_stats,
    script_dir: Optional[str] = None,
    edit_file=True,
    repro_test_info=None,
    regression_script_content: Optional[str] = None,
):
    instance_id = instance["instance_id"]
    base_dir = instance.get("base_dir", "/testbed")
    script_setup_path = None
    if regression_script_content is None:
        if not script_dir:
            raise ValueError("script_dir must be provided if regression_script_content is not provided")
        script_setup_path = get_bash_file_path(instance_id, base_dir=script_dir, regression=True)
    if repro_test_info is not None:
        ## test must be passed as a base64 encoded string: {instance_id: instance_id, test_patch: [test_patch_1, test_patch_2, ...]}
        reproduction_tests = json.loads(base64.b64decode(repro_test_info).decode())
    if not repro_test_info or not reproduction_tests:
        logging.warning("No reproduction tests found")
        return instance_stats

    # Define a regex pattern to match ANSI escape codes to remove color from output
    update_status = None
    try:
        # run setup script
        result = run_bash_script(script_path=script_setup_path, script_content=regression_script_content)
        logging.info("Script executed successfully.")
        content = format_output_stream(result)
        logging.info(content)

        # apply edits
        if edit_file:
            patch = None
            edit_file_path = None
            if "model_patch" in instance_stats[instance_id]:
                patch = instance_stats[instance_id]["model_patch"]
            else:
                raise ValueError("model_patch must be provided in instance_stats")

            update_status = update_files(edit_file_path, base_dir, patch)
            content += update_status
        else:
            update_status = ""

        reproduction_tests_results = []
        # run reproduction tests, I am using for loop for now.
        # TODO: improve to multiprocessing. not much improvement expected?
        for test_patch in reproduction_tests["test_patch"]:
            ## UPDATED: the index may not be the same as the order of the tests in the test_patch
            match = re.search(r"reproduce_bug_(\d+).py", test_patch)
            if match:
                index = int(match.group(1))
            else:
                raise ValueError(f"Could not find index in {test_patch}")
            result = run_reproduction_on_instance_single(make_reproduction_commands(test_patch), index)
            reproduction_tests_results.append(result)

    except subprocess.CalledProcessError as e:
        logging.error("Script execution failed.")
        out = e.stdout.replace("…", "...")
        err = e.stderr.replace("…", "...")
        content = out + err

    instance_stats[instance_id]["reproduction_tests_results"] = reproduction_tests_results
    instance_stats[instance_id]["log"] = content

    return instance_stats


def extract_test_exit_code(test_output):
    if "reproduction_tests_results" not in test_output:
        return []
    status_pattern = re.compile(r"reproduction test status:(\w+)")
    exit_codes = []
    for test_output in test_output["reproduction_tests_results"]:
        match = re.search(status_pattern, test_output["test_output"])
        if match:
            exit_codes.append(int(match.group(1)))
        else:
            exit_codes.append(-1)
    return exit_codes


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--script_dir",
        type=str,
        default=None,
        help="Directory of eval scripts (if not provided, use script_content instead)",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "repro-gen"],
        help="Choose between patch evaluation and bug reproduction and regression testing",
    )

    # Reproduction test info: allow either direct base64 string or a path to a file
    parser.add_argument(
        "--repro_test_info",
        type=str,
        default=None,
        help=(
            "Reproduction test info (a base64 encoded string: "
            "{instance_id: instance_id, test_patch: [test_patch_1, test_patch_2, ...]})"
        ),
    )
    parser.add_argument(
        "--repro_test_info_file",
        type=str,
        default=None,
        help="Path to a file containing base64-encoded repro_test_info.",
    )

    # Instance info: allow either direct base64 string or a path to a file
    parser.add_argument(
        "--instance_info",
        type=str,
        required=False,
        help=(
            "Instance info (a base64 encoded string of a dictionary with keys: "
            "instance_id,repo,setup_script,test_script,regression_script,"
            "PASS_TO_PASS,FAIL_TO_PASS,patch)"
        ),
    )
    parser.add_argument(
        "--instance_info_file",
        type=str,
        default=None,
        help="Path to a file containing base64-encoded instance_info.",
    )

    # Inference results: allow either direct base64 string or a path to a file
    parser.add_argument(
        "--inference_results",
        type=str,
        required=False,
        help=("Inference results (a base64 encoded string of a dictionary with keys: instance_id,model_patch}"),
    )
    parser.add_argument(
        "--inference_results_file",
        type=str,
        default=None,
        help="Path to a file containing base64-encoded inference_results.",
    )

    args = parser.parse_args()

    # Backwards-compatible validation: require either the direct string or file
    # variants for required payloads.
    if not args.instance_info and not args.instance_info_file:
        parser.error("One of --instance_info or --instance_info_file is required.")
    if not args.inference_results and not args.inference_results_file:
        parser.error("One of --inference_results or --inference_results_file is required.")

    return args


def main():
    args = parse_arguments()

    # Resolve inference_results payload
    if args.inference_results_file:
        with open(args.inference_results_file, "r") as f:
            inference_results_b64 = f.read()
    else:
        inference_results_b64 = args.inference_results
    inference_stats = json.loads(base64.b64decode(inference_results_b64).decode())

    # Resolve instance_info payload
    if args.instance_info_file:
        with open(args.instance_info_file, "r") as f:
            instance_info_b64 = f.read()
    else:
        instance_info_b64 = args.instance_info

    instance = json.loads(base64.b64decode(instance_info_b64).decode())
    edit_file = True
    setup_script_content = instance["setup_script"]
    test_script_content = instance["test_script"]
    regression_script_content = instance["regression_script"]
    instance_stats = {instance["instance_id"]: {**inference_stats}}

    if args.mode == "eval":
        res_instance_stats = run_evaluation_on_instance(
            instance,
            instance_stats,
            script_dir=args.script_dir,
            edit_file=edit_file,
            setup_script_content=setup_script_content,
            test_script_content=test_script_content,
        )
    elif args.mode == "repro-gen":
        status_pattern = re.compile(r"reproduction test status:(\w+)")
        # Resolve repro_test_info payload (optional)
        if args.repro_test_info_file and not args.repro_test_info:
            with open(args.repro_test_info_file, "r") as f:
                repro_test_info_b64 = f.read()
        else:
            repro_test_info_b64 = args.repro_test_info

        res_instance_stats = run_reproduction_on_instance(
            instance,
            deepcopy(instance_stats),
            script_dir=args.script_dir,
            repro_test_info=repro_test_info_b64,
            edit_file=False,
            regression_script_content=regression_script_content,
        )
        return_codes_before_patch = []
        for i in range(len(res_instance_stats[instance["instance_id"]]["reproduction_tests_results"])):
            match = re.search(
                status_pattern,
                res_instance_stats[instance["instance_id"]]["reproduction_tests_results"][i]["test_output"],
            )
            if match:
                return_codes_before_patch.append(int(match.group(1)))
        if "model_patch" not in instance_stats[instance["instance_id"]]:
            instance_stats[instance["instance_id"]]["model_patch"] = instance["patch"]
        res_instance_stats = run_reproduction_on_instance(
            instance,
            deepcopy(instance_stats),
            script_dir=args.script_dir,
            repro_test_info=repro_test_info_b64,
            edit_file=True,
            regression_script_content=regression_script_content,
        )
        return_codes_after_patch = []
        for i in range(len(res_instance_stats[instance["instance_id"]]["reproduction_tests_results"])):
            match = re.search(
                status_pattern,
                res_instance_stats[instance["instance_id"]]["reproduction_tests_results"][i]["test_output"],
            )
            if match:
                return_codes_after_patch.append(int(match.group(1)))
        print(f"[Return codes before patch]: {return_codes_before_patch}")
        print(f"[Return codes after patch]: {return_codes_after_patch}")
    else:
        raise ValueError(f"Mode {args.mode} not supported")

    if args.mode == "eval":
        print(
            res_instance_stats[instance["instance_id"]]["resolution"]
            if "resolution" in res_instance_stats[instance["instance_id"]]
            else "RESOLVED_NO"
        )
    return (
        res_instance_stats[instance["instance_id"]]["resolution"]
        if "resolution" in res_instance_stats[instance["instance_id"]]
        else "RESOLVED_NO"
    )


if __name__ == "__main__":
    main()
