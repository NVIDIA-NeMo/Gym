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
"""
This file contains patches copied from https://github.com/SWE-bench/SWE-bench/pull/630
"""

import re
from typing import Any, Dict, Optional

# @bxyu-nvidia: We import wildcard because there are a million imports otherwise...
from swebench.harness.constants import END_TEST_OUTPUT, FAIL_ONLY_REPOS, MAP_REPO_TO_EXT, START_TEST_OUTPUT
from swebench.harness.run_evaluation import *

from nemo_gym import PARENT_DIR
from nemo_gym.sandbox import AsyncSandbox
from resources_servers.swebench.failure_only_grading import failure_only_suite_completed
from resources_servers.swebench.row_evaluation import RowTestSpec, get_row_eval_report, karma_runs_completed


# The eval script's stderr joins its stdout, in order. SWE-bench's start and end markers are no-op commands that
# reach the output only through bash's trace, on stderr, and the log parser reads the region between them. A
# provider that returns a command's stdout and stderr apart (OpenSandbox, E2B) would otherwise put everything the
# tests print on stdout before the start marker.
EVAL_SCRIPT_COMMAND = "/bin/bash /eval.sh 2>&1"

# How SWE-bench 5 applies a patch (run_evaluation.py, v5.0.1, MIT): a three-way merge before the partial --reject,
# patch(1) only forwards, and a clean tree before each attempt. Used for rows that define their evaluation; the
# harness path keeps the pinned harness's GIT_APPLY_CMDS.
ROW_GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]
RESET_WORKTREE_COMMAND = "/bin/bash -c 'git checkout -- . ; git clean -fd'"


def eval_report_for(test_spec: TestSpec | RowTestSpec, pred: dict, test_output_path: Path) -> dict:
    """SWE-bench's report for ``pred``. A row that defines its own evaluation is graded with its own log parser and
    eval_type under SWE-bench 5's rules, which also refuse a fail_only verdict from a log that shows no sign the
    suite ran; any other test spec with the pinned harness."""
    if isinstance(test_spec, RowTestSpec):
        return get_row_eval_report(test_spec, pred, str(test_output_path), include_tests_status=True)
    return get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=test_output_path,
        include_tests_status=True,
    )


# @bxyu-nvidia: We modify the run_instance function to:
# 1. Enable async container operations.
# 2. Avoid complicated patching logic.
async def run_instance(
    test_spec: TestSpec | RowTestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
) -> dict:
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = eval_report_for(test_spec, pred, test_output_path)
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }

    if not test_spec.is_remote_image:
        # Link the image build dir in the log dir
        build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
        image_build_link = log_dir / "image_build_dir"
        if not image_build_link.exists():
            try:
                # link the image build dir in the log dir
                image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
            except:
                # some error, idk why
                pass

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    eval_completed = False
    report = {}
    # Why the evaluation did not complete: the model's patch could not be applied, and the error that ended it.
    patch_apply_failed = False
    error = None
    try:
        # Original code:
        # Build + start instance container (instance image should already be built)
        # container = build_container(
        #     test_spec, client, run_id, logger, rm_image, force_rebuild
        # )
        # container.start()
        # logger.info(f"Container for {instance_id} started: {container.id}")
        #
        # Modified code:
        container = client  # We just directly pass the Docker wrapper in.

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container...")
        # Original code:
        # copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))
        #
        # Modified code:
        await container.copy(patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container (TODO: FIX THIS)
        applied_patch = False
        row_defined = isinstance(test_spec, RowTestSpec)
        for attempt, git_apply_cmd in enumerate(ROW_GIT_APPLY_CMDS if row_defined else GIT_APPLY_CMDS):
            if row_defined and attempt:
                # A failed attempt (notably --reject) leaves a partly patched tree; start the next from a clean one.
                await container.exec_run(RESET_WORKTREE_COMMAND, workdir=DOCKER_WORKDIR, user=DOCKER_USER)
            val = await container.exec_run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch and row_defined:
            # The fallbacks can leave the patch fully applied while each exited non-zero.
            reverse_check = await container.exec_run(
                f"git apply --check --reverse {DOCKER_PATCH}", workdir=DOCKER_WORKDIR, user=DOCKER_USER
            )
            if reverse_check.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}: verified already applied")
                applied_patch = True

        # The jqlang__jq-2681 image contains dirty generated lexer files from
        # a different Flex version. Apply its golden source files only; jq's
        # evaluation build regenerates lexer.c/.h and parser.c/.h from them.
        if not applied_patch and instance_id == "jqlang__jq-2681":
            val = await container.exec_run(
                f"git apply --include=src/lexer.l --include=src/parser.y {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}: applied source patch before regeneration\n{val.output.decode(UTF8)}")
                applied_patch = True

        if not applied_patch:
            patch_apply_failed = True
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                logger,
            )

        # Get git diff before running eval script
        git_diff_output_before = (
            (await container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR))
            .output.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(f"Eval script for {instance_id} written to {eval_file}; copying to container...")
        # Original code:
        # copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
        #
        # Modified code:
        await container.copy(eval_file, PurePosixPath("/eval.sh"))

        # Run eval script, write output to logs
        # Original code:
        # test_output, timed_out, total_runtime = exec_run_with_timeout(
        #     container, "/bin/bash /eval.sh", timeout
        # )
        #
        # Modified code:
        test_output, timed_out, total_runtime = await container.exec_run_with_timeout(
            EVAL_SCRIPT_COMMAND, timeout=timeout
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get git diff after running eval script (ignore permission changes)
        git_diff_output_after = (
            (await container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR))
            .output.decode(UTF8)
            .strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        eval_report = eval_report_for(test_spec, pred, test_output_path)
        # Failure-only grading counts every test missing from the log as passing, so a suite that never started or
        # stopped part way would read as resolved. Accept that verdict only from a suite that ran to completion. The
        # row grader makes its own check.
        if (
            eval_report[instance_id]["resolved"]
            and not isinstance(test_spec, RowTestSpec)
            and test_spec.repo in FAIL_ONLY_REPOS
            and not failure_only_suite_completed(test_spec.repo, test_output)
        ):
            raise EvaluationError(
                instance_id,
                f"The {test_spec.repo} test suite did not run to completion, so its failure-only report shows no "
                "verdict",
                logger,
            )
        # The same rule for a fail_only row: a Karma run that stopped part way shows no verdict.
        if (
            eval_report[instance_id]["resolved"]
            and isinstance(test_spec, RowTestSpec)
            and test_spec.eval_type == "fail_only"
            and not karma_runs_completed(test_output)
        ):
            raise EvaluationError(
                instance_id,
                f"The {test_spec.repo} Karma run did not execute every planned test, so its failure-only report shows "
                "no verdict",
                logger,
            )
        report = eval_report
        logger.info(f"report: {report}\nResult for {instance_id}: resolved: {report[instance_id]['resolved']}")

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except (EvaluationError, BuildImageError) as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
        error = f"{type(e).__name__}: {e}"
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
        error = f"{type(e).__name__}: {e}"
    finally:
        # Original code:
        # Remove instance container + image, close logger
        # cleanup_container(client, container, logger)
        #
        # Modified code:
        await container.cleanup()

        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
        return {
            "completed": eval_completed,
            "resolved": report.get(instance_id, {}).get("resolved", False),
            "patch_apply_failed": patch_apply_failed,
            "error": error,
        }


# OpenLayers' tests use WebGL in Chrome under Xvfb. Without a GPU that needs Chrome's software renderer (SwiftShader),
# which Chrome does not fall back to in a sandbox that lets it start its own sandbox: WebGL context creation fails and
# a WebGL test's hook error ends the Karma suite part way. Under Docker's default seccomp Chrome cannot start its
# sandbox and runs unsandboxed with software WebGL (SWE-bench#636). A wrapper at the path the evaluation script launches
# Chrome from gives it that setup; with fewer flags, Karma still stopped part way in some runs.
OPENLAYERS_CHROME_FLAGS = "--no-sandbox --enable-unsafe-swiftshader --use-angle=swiftshader --ignore-gpu-blocklist"
OPENLAYERS_CHROME_WRAPPER = (
    "[ -e /usr/bin/google-chrome-stable.real ] || { mv /usr/bin/google-chrome-stable /usr/bin/google-chrome-stable.real"
    f" && printf '#!/bin/sh\\nexec /usr/bin/google-chrome-stable.real {OPENLAYERS_CHROME_FLAGS} \"$@\"\\n'"
    " > /usr/bin/google-chrome-stable && chmod 755 /usr/bin/google-chrome-stable; }"
)


def patch_swebench_openlayers_software_webgl(test_spec: Any) -> None:
    """Give OpenLayers' Chrome the software WebGL renderer before its tests start."""
    if test_spec.repo != "openlayers/openlayers" or OPENLAYERS_CHROME_WRAPPER in test_spec.eval_script_list:
        return
    start = f": '{START_TEST_OUTPUT}'"
    if start not in test_spec.eval_script_list:
        raise ValueError(f"The evaluation script of {test_spec.instance_id} has no start-of-test-output marker")
    test_spec.eval_script_list.insert(test_spec.eval_script_list.index(start), OPENLAYERS_CHROME_WRAPPER)


# Some evaluation scripts run several `xvfb-run` commands on display :99 one after another. xvfb-run returns before the
# X server it stopped has exited, so the next one can fail with "Xvfb failed to start" and that suite never runs. A
# wrapper makes xvfb-run wait (up to 30 s) until the server named in the display's lock file has exited.
XVFB_RUN_WRAPPER = (
    "[ -e /usr/bin/xvfb-run.real ] || { mv /usr/bin/xvfb-run /usr/bin/xvfb-run.real && printf '#!/bin/sh\\n"
    'for i in $(seq 30); do [ -e /tmp/.X99-lock ] && kill -0 "$(tr -d " " < /tmp/.X99-lock)" 2>/dev/null '
    '|| break; sleep 1; done\\nexec /usr/bin/xvfb-run.real "$@"\\n\' > /usr/bin/xvfb-run && chmod 755 /usr/bin/xvfb-run; }'
)


def patch_swebench_xvfb_run_wait(test_spec: Any) -> None:
    """Make a script that starts Xvfb wait for the previous X server before starting the next."""
    start = f": '{START_TEST_OUTPUT}'"
    if XVFB_RUN_WRAPPER in test_spec.eval_script_list or start not in test_spec.eval_script_list:
        return
    index = test_spec.eval_script_list.index(start)
    if any("xvfb-run" in line for line in test_spec.eval_script_list[index:]):
        test_spec.eval_script_list.insert(index, XVFB_RUN_WRAPPER)


########################################
# START SWE Bench Verified instance patches
########################################


def patch_swebench_verified_env_vars(repo: str, env: Dict[str, Any]) -> None:
    if MAP_REPO_TO_EXT.get(repo) == "py":
        # Activate the SWEBench Verified python conda environment
        env["CONDA_EXE"] = "/opt/miniconda3/bin/conda"
        env["_CE_M"] = ""
        env["CONDA_PREFIX"] = "/opt/miniconda3/envs/testbed"
        env["CONDA_PROMPT_MODIFIER"] = "(testbed) "
        env["_CE_CONDA"] = ""
        env["CONDA_SHLVL"] = "2"
        env["SHLVL"] = "0"
        env["CONDA_PYTHON_EXE"] = "/opt/miniconda3/bin/python"
        env["CONDA_DEFAULT_ENV"] = "testbed"
        env["PATH"] = (
            "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/condabin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        env["CONDA_PREFIX_1"] = "/opt/miniconda3"


# Make every pytest run in a Python evaluation script print the per-test result lines (-rA) the log parsers read. Some
# repos run pytest through another tool without the flag: a SWE-bench Verified Sphinx run goes through tox, whose
# commands line has no -rA when the image's tox.ini edit is missing. tox passes PYTEST_ADDOPTS on to pytest.
PYTEST_RESULT_SUMMARY = 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"'


def patch_swebench_pytest_result_summary(test_spec: TestSpec) -> None:
    if MAP_REPO_TO_EXT.get(test_spec.repo) != "py" or PYTEST_RESULT_SUMMARY in test_spec.eval_script_list:
        return
    start = f": '{START_TEST_OUTPUT}'"
    if start not in test_spec.eval_script_list:
        raise ValueError(f"The evaluation script of {test_spec.instance_id} has no start-of-test-output marker")
    test_spec.eval_script_list.insert(test_spec.eval_script_list.index(start), PYTEST_RESULT_SUMMARY)


# Chart.js runs its Karma suite in Chrome, which keeps renderer shared memory in /dev/shm. A container's default 64 MB
# /dev/shm is too small for the suite's 1,100-1,600 canvas tests: Chrome crashes part way, Karma never prints its
# summary, and the failure-only completion check rejects the run, the gold patch's included. A sandbox cannot enlarge
# /dev/shm without CAP_SYS_ADMIN, so Chrome is told to keep that memory under /tmp instead.
CHARTJS_CHROME_FLAG = "--disable-dev-shm-usage"
_KARMA_CONFIG = re.compile(r"karma start \./(karma\.conf\.c?js)\b")


def chartjs_chrome_flag_edit(karma_config: str) -> str:
    """The command that adds ``CHARTJS_CHROME_FLAG`` to the flags of the Chrome launcher in ``karma_config``."""
    return f"sed -i \"/flags: \\[/a\\          '{CHARTJS_CHROME_FLAG}',\" ./{karma_config}"


def patch_swebench_chartjs_chrome_flag(test_spec: TestSpec) -> None:
    # SWE-bench runs Chart.js 2.x through Gulp, without a direct Karma invocation.
    if test_spec.repo != "chartjs/Chart.js" or test_spec.version.startswith("2."):
        return
    for index, line in enumerate(test_spec.eval_script_list):
        karma_config = _KARMA_CONFIG.search(line)
        if karma_config is None:
            continue
        edit = chartjs_chrome_flag_edit(karma_config.group(1))
        if edit not in test_spec.eval_script_list:
            test_spec.eval_script_list.insert(index, edit)
        return
    raise ValueError(f"The evaluation script of {test_spec.instance_id} does not start Karma")


########################################
# START SWE Bench Multilingual instance patches
########################################

# @bxyu-nvidia: These are the patches to the `eval.sh` produced by SWE Bench that needed to be done in order for the golden patches to pass
# Each patch here is not intended to help the model, they are literally making the test cases runnable.
# Most of these are specific to Nvidia's OpenSandbox server. A lot of these aren't issues when run on bare metal AWS EC2 instances.
# These patches may or may not be relevant to your specific sandboxing setup.


def patch_swebench_multilingual_golden_patch_pass(eval_sh: str, instance_id: str) -> str:
    # Run Maven tests without the daemon which causes issues with gson tests.
    eval_sh = eval_sh.replace("mvnd test", "mvn test")

    # apache__druid-16875 never reaches its focused tests because the
    # build-metadata plugin's `git describe` exceeds its 30s timeout.
    if instance_id == "apache__druid-16875":
        eval_sh = eval_sh.replace("mvn test", "mvn test -Dgit.commit.id.skip=true")

    # These projects resolve Gradle plugins through Plugin Portal. Load the
    # uploaded mirror script explicitly because the sandbox Gradle process does
    # not auto-discover init.d under /root.
    if instance_id in {"apache__lucene-13494", "reactivex__rxjava-7597"}:
        eval_sh = eval_sh.replace(
            "./gradlew", "./gradlew --init-script /root/.gradle/init.d/maven_central_mirror.gradle"
        )

    # axios__axios-4738 needs more than 10s for cold dependency and process startup.
    eval_sh = eval_sh.replace("timeout 10s", "timeout 120s")

    # Preact's Chrome tests use a 2s Mocha timeout, which is too short
    if "preactjs__preact" in instance_id:
        eval_sh = eval_sh.replace(
            "npx karma start karma.conf.js", "npx karma start karma.conf.js --client.mocha.timeout=60000"
        )

    # valkey-io__valkey-928 checks the source node immediately after an
    # asynchronous replica migration. Let the cluster state settle before the
    # test's role assertion.
    if instance_id == "valkey-io__valkey-928":
        eval_sh = eval_sh.replace(
            "TERM=dumb ./runtest",
            "sed -i 's/assert_equal \\[lindex \\[R 3 role\\] 2\\] {}/after 5000; assert_equal [lindex [R 3 role] 2] {}/' "
            "tests/unit/cluster/replica-migration.tcl\nTERM=dumb ./runtest",
        )

    return eval_sh


async def patch_swebench_multilingual_sandbox(
    repo: str, instance_id: str, sandbox: AsyncSandbox, *, overwrite_maven_settings: bool = True
) -> None:
    if MAP_REPO_TO_EXT.get(repo) == "java":
        base_path = PARENT_DIR / "responses_api_agents/swe_agents/maven_mirror"
        settings_xml_path = base_path / "settings.xml"
        init_gradle_path = base_path / "init.gradle"

        # This init.d is necessary for some Java tests to properly pull from the maven mirror
        # e.g. apache__lucene and apache__druid
        #
        # Lucene's applied Gradle scripts have their own buildscript scopes. Those scopes
        # are not exposed through the root project's repository handler, so an init script
        # cannot rewrite them before they resolve. Rewrite Maven Central references in all
        # checked-in Gradle scripts before Gradle starts (but never mutate its cache).
        await sandbox.exec("""mkdir -p /root/.m2 ~/.gradle/init.d \
        && if [ -d gradle ]; then
find . -path './.gradle' -prune -o -type f \\( -name '*.gradle' -o -name '*.gradle.kts' \\) -exec sed -i 's#mavenCentral()#maven { url = uri("https://maven-central.storage-download.googleapis.com/maven2/") }#g; s#https://repo.maven.apache.org/maven2#https://maven-central.storage-download.googleapis.com/maven2#g; s#https://repo1.maven.org/maven2#https://maven-central.storage-download.googleapis.com/maven2#g' {} +
fi""")

        # This settings.xml is necessary for some Java tests to properly pull from the maven mirror
        if overwrite_maven_settings:
            await sandbox.upload(settings_xml_path, "/root/.m2/settings.xml")

        # This init.d is necessary for some Java tests to properly pull from the maven mirror
        await sandbox.upload(init_gradle_path, "/root/.gradle/init.d/maven_central_mirror.gradle")

    # tokio-rs__tokio-4384 otherwise resolves getrandom 0.4.3, which
    # requires Cargo 1.85 while its image provides Cargo 1.81.
    if instance_id == "tokio-rs__tokio-4384":
        await sandbox.exec(
            "cargo update -p getrandom@0.4.3 --precise 0.4.2 && cargo update -p proptest@1.11.0 --precise 1.5.0"
        )


def patch_swebench_multilingual_log_parsing(stdout: str, stderr: str, instance_id: str) -> Optional[str]:
    test_output = None

    # Only for a command whose stderr comes back apart; EVAL_SCRIPT_COMMAND joins the streams itself.
    # For RuboCop tests in SWE Multilingual specifically, there is an issue with the logs parsing if the stdout and stderr returned is not interleaved.
    # We interleave the stderr inside the start and end tags in the stdout here instead. See `get_logs_eval`
    if "rubocop" in instance_id and START_TEST_OUTPUT in stderr:
        start, middle_end = stderr.split(START_TEST_OUTPUT)
        middle, end = middle_end.split(END_TEST_OUTPUT)
        test_output = start + START_TEST_OUTPUT + stdout + middle + END_TEST_OUTPUT + end

    return test_output
