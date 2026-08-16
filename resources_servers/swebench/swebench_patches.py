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

from typing import Any, Dict

# @bxyu-nvidia: We import wildcard because there are a million imports otherwise...
from swebench.harness.constants import MAP_REPO_TO_EXT
from swebench.harness.run_evaluation import *

from nemo_gym import PARENT_DIR
from nemo_gym.sandbox import AsyncSandbox


# @bxyu-nvidia: We modify the run_instance function to:
# 1. Enable async container operations.
# 2. Avoid complicated patching logic.
async def run_instance(
    test_spec: TestSpec,
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
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
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
        for git_apply_cmd in GIT_APPLY_CMDS:
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
            "/bin/bash /eval.sh", timeout=timeout
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
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(f"report: {report}\nResult for {instance_id}: resolved: {report[instance_id]['resolved']}")

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except (EvaluationError, BuildImageError) as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
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
        }


########################################
# START SWE Bench Multilingual instance patches
########################################

# @bxyu-nvidia: These are the patches to the `eval.sh` produced by SWE Bench that needed to be done in order for the golden patches to pass
# Each patch here is not intended to help the model, they are literally making the test cases runnable.
# Most of these are specific to Nvidia's OpenSandbox server. A lot of these aren't issues when run on bare metal AWS EC2 instances.
# These patches may or may not be relevant to your specific sandboxing setup.


def patch_for_swebench_multilingual_golden_patch_pass(eval_sh: str, instance_id: str) -> str:
    # This init.d is necessary for some Java tests to properly pull from the maven mirror
    # e.g. apache__lucene and apache__druid
    #
    # Lucene's applied Gradle scripts have their own buildscript scopes. Those scopes
    # are not exposed through the root project's repository handler, so an init script
    # cannot rewrite them before they resolve. Rewrite Maven Central references in all
    # checked-in Gradle scripts before Gradle starts (but never mutate its cache).
    lucene_mirror_setup = """if [ -d gradle ]; then
find . -path './.gradle' -prune -o -type f \\( -name '*.gradle' -o -name '*.gradle.kts' \\) -exec sed -i 's#mavenCentral()#maven { url = uri("https://maven-central.storage-download.googleapis.com/maven2/") }#g; s#https://repo.maven.apache.org/maven2#https://maven-central.storage-download.googleapis.com/maven2#g; s#https://repo1.maven.org/maven2#https://maven-central.storage-download.googleapis.com/maven2#g' {} +
fi
./gradlew --init-script /root/.gradle/init.d/maven_central_mirror.gradle test"""
    data = data.replace("./gradlew test", lucene_mirror_setup)

    # Run Maven tests without the daemon which causes issues with gson tests.
    data = data.replace("mvnd test", "mvn test")

    # apache__druid-16875 never reaches its focused tests because the
    # build-metadata plugin's `git describe` exceeds its 30s timeout.
    if instance_id == "apache__druid-16875":
        data = data.replace("mvn test", "mvn test -Dgit.commit.id.skip=true")

    # valkey-io__valkey-928 checks the source node immediately after
    # an asynchronous replica migration. Allow the cluster state to
    # settle before its role assertion runs.
    if instance_id == "valkey-io__valkey-928":
        data = data.replace(
            "TERM=dumb ./runtest",
            "sed -i 's/assert_equal \\[lindex \\[R 3 role\\] 2\\] {}/after 5000; assert_equal [lindex [R 3 role] 2] {}/' "
            "tests/unit/cluster/replica-migration.tcl\nTERM=dumb ./runtest",
        )

    # axios__axios-4738 needs more than 10s for cold dependency and process startup.
    data = data.replace("timeout 10s", "timeout 120s")

    # tokio-rs__tokio-4384 otherwise resolves getrandom 0.4.3, which
    # requires Cargo 1.85 while its image provides Cargo 1.81.
    if instance_id == "tokio-rs__tokio-4384":
        data = data.replace(
            "RUSTFLAGS=-Awarnings cargo test",
            "cargo update -p getrandom@0.4.3 --precise 0.4.2 && cargo update -p proptest@1.11.0 --precise 1.5.0 && RUSTFLAGS=-Awarnings cargo test",
        )

    # Preact's Chrome tests use a 2s Mocha timeout, which is too short
    if "preactjs__preact" in instance_id:
        data = data.replace(
            "npx karma start karma.conf.js", "npx karma start karma.conf.js --client.mocha.timeout=60000"
        )

    return data


def patch_swebench_multilingual_resources_request(resources: Dict[str, Any], instance_id: str) -> None:
    # Chrome is OOM-killed before Karma can connect for preactjs__preact-
    # {2896,4316,4436}; reserve enough memory for its two-browser runner.
    if instance_id in {"preactjs__preact-2896", "preactjs__preact-4316", "preactjs__preact-4436"}:
        resources["memory_mib"] = max(resources.get("memory_mib", 0), 16 * 1024)


async def patch_swebench_multilingual_sandbox_upload(repo: str, sandbox: AsyncSandbox) -> None:
    if MAP_REPO_TO_EXT.get(repo) == "java":
        base_path = PARENT_DIR / "responses_api_agents/swe_agents/maven_mirror"
        settings_xml_path = base_path / "settings.xml"
        init_gradle_path = base_path / "init.gradle"

        await sandbox.exec("""mkdir -p /root/.m2 /root/.gradle/init.d""")

        # This settings.xml is necessary for some Java tests to properly pull from the maven mirror
        await sandbox.upload(settings_xml_path, "/root/.m2/settings.xml")

        # This init.d is necessary for some Java tests to properly pull from the maven mirror
        await sandbox.upload(init_gradle_path, "/root/.gradle/init.d/maven_central_mirror.gradle")
