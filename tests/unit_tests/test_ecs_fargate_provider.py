# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ECS Fargate provider tests — all AWS/SSH/network calls mocked."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import (
    SandboxSpec,
    SandboxStatus,
    create_provider,
    get_provider_class,
    list_providers,
)
from nemo_gym.sandbox.providers.ecs_fargate import EcsFargateProvider, engine
from nemo_gym.sandbox.providers.ecs_fargate.provider import _apply_spec_overrides, engine_config_from_mapping


_ENG = "nemo_gym.sandbox.providers.ecs_fargate.engine"


def _provider_config(**overrides):
    cfg = dict(
        region="us-west-2",
        cluster="test-cluster",
        subnets=["subnet-aaa"],
        security_groups=["sg-bbb"],
        assign_public_ip=True,
        execution_role_arn="arn:aws:iam::1234:role/ecsTaskExec",
        task_role_arn="arn:aws:iam::1234:role/ecsTask",
        ssh_sidecar={
            "sshd_port": 2222,
            "public_key_secret_arn": "arn:aws:secretsmanager:us-east-1:1234:secret:pub",  # pragma: allowlist secret
            "private_key_secret_arn": "arn:aws:secretsmanager:us-east-1:1234:secret:priv",  # pragma: allowlist secret
            "exec_server_port": 5000,
        },
    )
    cfg.update(overrides)
    return {"ecs_fargate": cfg}


@contextlib.contextmanager
def _mock_engine_start(exec_result=None):
    """Patch every AWS/SSH seam so ``EcsFargateSandbox.start`` runs offline.

    Yields the fake exec client so delegation can be asserted.
    """
    tunnel = MagicMock()
    tunnel.local_port = 19000
    exec_client = MagicMock()
    exec_client.exec = AsyncMock(return_value=exec_result or engine.ExecResult("out", "err", 0))
    exec_client.upload = AsyncMock()
    exec_client.download = AsyncMock(return_value=b"payload")
    exec_client.close = AsyncMock()

    with (
        patch.object(engine.EcsFargateSandbox, "_init_aws_clients"),
        patch.object(engine.EcsFargateSandbox, "_resolve_image", return_value="python:3.12"),
        patch.object(engine.EcsFargateSandbox, "_register_task_definition", return_value="task-def-arn"),
        patch.object(engine.EcsFargateSandbox, "_run_task", return_value="task-arn"),
        patch.object(engine.EcsFargateSandbox, "_register_for_cleanup"),
        patch.object(engine.EcsFargateSandbox, "_wait_for_running"),
        patch.object(engine.EcsFargateSandbox, "_get_task_public_ip", return_value="10.0.0.10"),
        patch.object(engine.EcsFargateSandbox, "_wait_for_ssh_ready"),
        patch(f"{_ENG}.download_secret_to_file", return_value="/tmp/key"),
        patch(f"{_ENG}.download_secret_to_string", return_value="ssh-rsa fake"),
        patch(f"{_ENG}.build_ssh_sidecar_container", return_value={"name": "ssh-tunnel"}),
        patch(f"{_ENG}._free_port", return_value=19001),
        patch(f"{_ENG}.SshTunnel", return_value=tunnel) as tunnel_cls,
        patch(f"{_ENG}.ExecClient", return_value=exec_client),
    ):
        yield exec_client, tunnel_cls


# ── Registration ──────────────────────────────────────────────────────


def test_provider_registered():
    assert "ecs_fargate" in list_providers()
    assert get_provider_class("ecs_fargate") is EcsFargateProvider
    assert EcsFargateProvider.name == "ecs_fargate"


# ── Config resolution ─────────────────────────────────────────────────


def test_config_explicit_values_no_ssm():
    p = create_provider(_provider_config())
    cfg = p._cfg
    assert cfg.cluster == "test-cluster"
    assert cfg.subnets == ["subnet-aaa"]
    assert cfg.assign_public_ip is True
    assert cfg.ssh_sidecar.exec_server_port == 5000
    assert cfg.ssh_sidecar.sshd_port == 2222


def test_config_ssm_autodiscovery_merges_and_yaml_wins():
    ssm_blob = {
        "cluster": "ssm-cluster",
        "subnets": ["subnet-ssm"],
        "security_groups": ["sg-ssm"],
        "execution_role_arn": "arn:ssm:exec",
        "ssh_sidecar": {
            "public_key_secret_arn": "arn:ssm:pub",  # pragma: allowlist secret
            "private_key_secret_arn": "arn:ssm:priv",  # pragma: allowlist secret
            "sshd_port": 52222,
        },
    }
    with patch(f"{_ENG}.resolve_ecs_config_from_ssm", return_value=ssm_blob) as resolve:
        # region set, cluster omitted -> SSM is consulted.
        p = create_provider({"ecs_fargate": {"region": "us-west-2", "subnets": ["subnet-override"]}})
    resolve.assert_called_once_with("us-west-2", "harbor")
    cfg = p._cfg
    assert cfg.cluster == "ssm-cluster"  # filled from SSM
    assert cfg.subnets == ["subnet-override"]  # explicit YAML wins
    assert cfg.execution_role_arn == "arn:ssm:exec"
    assert cfg.ssh_sidecar.public_key_secret_arn == "arn:ssm:pub"  # pragma: allowlist secret
    assert cfg.ssh_sidecar.sshd_port == 52222


def test_config_no_ssm_when_cluster_present():
    with patch(f"{_ENG}.resolve_ecs_config_from_ssm") as resolve:
        create_provider(_provider_config())
    resolve.assert_not_called()


def test_sidecar_missing_key_arns_raises():
    with pytest.raises(ValueError, match="public_key_secret_arn"):
        engine_config_from_mapping({"cluster": "c", "ssh_sidecar": {"sshd_port": 2222}})


# ── create / lifecycle delegation ─────────────────────────────────────


async def test_create_returns_handle_with_running_sandbox():
    provider = create_provider(_provider_config())
    spec = SandboxSpec(image="python:3.12")
    with _mock_engine_start():
        handle = await provider.create(spec)
    assert handle.provider_name == "ecs_fargate"
    assert handle.sandbox_id == "task-arn"
    assert isinstance(handle.raw, engine.EcsFargateSandbox)
    assert await provider.status(handle) == SandboxStatus.RUNNING
    await provider.close(handle)
    assert await provider.status(handle) == SandboxStatus.STOPPED


async def test_exec_maps_engine_result():
    provider = create_provider(_provider_config())
    spec = SandboxSpec(image="python:3.12", workdir="/work")
    with _mock_engine_start(exec_result=engine.ExecResult("hello\n", "", 0)) as (exec_client, _):
        handle = await provider.create(spec)
        result = await provider.exec(handle, "echo hello", timeout_s=42)
    assert (result.stdout, result.stderr, result.return_code) == ("hello\n", "", 0)
    # engine.exec received the gym timeout as timeout_sec
    _, kwargs = exec_client.exec.call_args
    assert kwargs["timeout"] == 42


async def test_upload_and_download_delegate(tmp_path):
    provider = create_provider(_provider_config())
    spec = SandboxSpec(image="python:3.12")
    src = tmp_path / "in.txt"
    src.write_text("data")
    dest = tmp_path / "out.txt"
    with _mock_engine_start() as (exec_client, _):
        handle = await provider.create(spec)
        await provider.upload_file(handle, src, "/remote/in.txt")
        await provider.download_file(handle, "/remote/out.txt", dest)
    exec_client.upload.assert_awaited_once()
    assert dest.read_bytes() == b"payload"


async def test_outside_endpoints_build_reverse_tunnel():
    provider = create_provider(_provider_config())
    spec = SandboxSpec(
        image="python:3.12",
        provider_options={"outside_endpoints": [{"url": "http://127.0.0.1:4000/v1", "env_var": "MODEL_BASE_URL"}]},
    )
    with _mock_engine_start() as (_, tunnel_cls):
        handle = await provider.create(spec)
    # exec-server mode opens a forward tunnel to the exec server + a reverse
    # tunnel for each outside endpoint.
    _, kwargs = tunnel_cls.call_args
    assert kwargs["forward_port"] == 5000
    # reverse spec format: "<remote_port>:<host>:<target_port>"
    assert any(s.endswith(":127.0.0.1:4000") for s in kwargs["reverses"])
    # the resolved endpoint is injected into the container env
    routing = handle.raw._outside_endpoint_routing
    assert routing.resolved_endpoint_url("MODEL_BASE_URL").startswith("http://127.0.0.1:")


async def test_create_missing_image_raises():
    provider = create_provider(_provider_config())
    from nemo_gym.sandbox.providers import SandboxCreateError

    with pytest.raises(SandboxCreateError, match="requires SandboxSpec.image"):
        await provider.create(SandboxSpec(image=None))


# ── Public AsyncSandbox surface ───────────────────────────────────────


async def test_async_sandbox_end_to_end():
    spec = SandboxSpec(image="python:3.12", files={"/app/run.sh": "echo hi"})
    with _mock_engine_start() as (exec_client, _):
        sb = AsyncSandbox(_provider_config(), spec)
        await sb.start()
        # initial files uploaded via the provider during start()
        exec_client.upload.assert_awaited()
        res = await sb.exec("ls")
        assert res.return_code == 0
        assert await sb.status() == SandboxStatus.RUNNING
        await sb.stop()
    assert await sb.status() == SandboxStatus.STOPPED


# ── Pure engine helpers (no AWS) ──────────────────────────────────────


def test_task_def_hash_ignores_log_config():
    base = {
        "family": "f1",
        "containerDefinitions": [
            {"name": "main", "image": "x", "logConfiguration": {"options": {"awslogs-group": "g1"}}}
        ],
    }
    other = {
        "family": "f2",
        "containerDefinitions": [
            {"name": "main", "image": "x", "logConfiguration": {"options": {"awslogs-group": "g2"}}}
        ],
    }
    assert engine._compute_task_def_hash(base) == engine._compute_task_def_hash(other)


def test_generate_buildspec_pushes_to_ecr():
    cfg = engine.EcsFargateConfig(
        region="us-west-2",
        ecr_repository="123.dkr.ecr.us-west-2.amazonaws.com/sandbox",
    )
    spec = engine.ImageBuilder._generate_buildspec(cfg, "sandbox", "tag1", f"{cfg.ecr_repository}:tag1")
    assert "docker build -t sandbox:tag1" in spec
    assert f"docker push {cfg.ecr_repository}:tag1" in spec


def _resolve_image_for(image, **cfg_overrides):
    cfg = engine.EcsFargateConfig(region="us-east-1", **cfg_overrides)
    sandbox = engine.EcsFargateSandbox(engine.SandboxSpec(image=image), ecs_config=cfg)
    return sandbox._resolve_image()


def test_resolve_image_routes_bare_name_to_ecr_mirror():
    # Bare/public names are mirrored to the ECR tag, never pulled directly.
    ecr = "463701203462.dkr.ecr.us-east-1.amazonaws.com/harbor-us-east-1"
    resolved = _resolve_image_for(
        "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest", ecr_repository=ecr
    )
    assert (
        resolved
        == f"{ecr}:{engine._sanitize_id('docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest')}"
    )
    assert "docker.io" not in resolved.split(":", 1)[1]  # origin registry not used for the pull


def test_resolve_image_passes_through_existing_ecr_ref():
    # A reference already in the ECR mirror is used verbatim (tag preserved,
    # including the double underscore the sanitizer would otherwise collapse).
    ecr = "463701203462.dkr.ecr.us-east-1.amazonaws.com/harbor-us-east-1"
    existing = f"{ecr}:nel-harbor-tasks-swe-bench-astropy-1-1ccf0d50cb33__1ccf0d50"
    assert _resolve_image_for(existing, ecr_repository=ecr) == existing


def test_resolve_image_template_takes_precedence():
    ecr = "463701203462.dkr.ecr.us-east-1.amazonaws.com/harbor-us-east-1"
    resolved = _resolve_image_for("anything", ecr_repository=ecr, image_template="{task_id}-built")
    assert resolved == "anything-built"


def test_is_ecr_image_ref_matches_only_ecr_hosts():
    assert engine._is_ecr_image_ref("463701203462.dkr.ecr.us-east-1.amazonaws.com/repo:tag")
    assert not engine._is_ecr_image_ref("docker.io/swebench/sweb.eval:latest")
    assert not engine._is_ecr_image_ref("ubuntu:24.04")


def test_generate_mirror_buildspec_pulls_tags_and_pushes():
    cfg = engine.EcsFargateConfig(
        region="us-east-1",
        ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/mirror",
        dockerhub_secret_arn="arn:aws:secretsmanager:us-east-1:123:secret:dh",  # pragma: allowlist secret
    )
    src = "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    ecr_url = f"{cfg.ecr_repository}:{engine._sanitize_id(src)}"
    spec = engine.ImageBuilder._generate_mirror_buildspec(cfg, src, ecr_url)
    assert f"docker pull --platform linux/amd64 {src}" in spec
    assert f"docker tag {src} {ecr_url}" in spec
    assert f"docker push {ecr_url}" in spec
    assert "get-login-password" in spec  # ECR login
    assert "secretsmanager get-secret-value" in spec  # Docker Hub login


def test_ensure_mirrored_skips_when_already_present():
    cfg = engine.EcsFargateConfig(region="us-east-1", ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/mirror")
    with (
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=True),
        patch.object(engine.ImageBuilder, "run_buildspec_via_codebuild") as cb,
    ):
        url = engine.ImageBuilder.ensure_mirrored(cfg=cfg, src_image="ubuntu:24.04")
    cb.assert_not_called()
    assert url == f"{cfg.ecr_repository}:{engine._sanitize_id('ubuntu:24.04')}"


def test_ensure_mirrored_runs_codebuild_when_missing():
    cfg = engine.EcsFargateConfig(region="us-east-1", ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/mirror")
    with (
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=False),
        patch.object(engine.ImageBuilder, "run_buildspec_via_codebuild") as cb,
    ):
        url = engine.ImageBuilder.ensure_mirrored(cfg=cfg, src_image="ubuntu:24.04")
    cb.assert_called_once()
    assert url.endswith(engine._sanitize_id("ubuntu:24.04"))


async def test_create_auto_mirrors_missing_public_image():
    ecr = "123.dkr.ecr.us-east-1.amazonaws.com/mirror"
    provider = create_provider(_provider_config(ecr_repository=ecr))
    spec = SandboxSpec(image="docker.io/swebench/sweb.eval:latest")
    with _mock_engine_start(), patch.object(engine.ImageBuilder, "ensure_mirrored") as m:
        await provider.create(spec)
    m.assert_called_once()
    assert m.call_args.kwargs["src_image"] == "docker.io/swebench/sweb.eval:latest"


async def test_create_skips_mirror_for_existing_ecr_ref():
    ecr = "463701203462.dkr.ecr.us-east-1.amazonaws.com/mirror"
    provider = create_provider(_provider_config(ecr_repository=ecr))
    spec = SandboxSpec(image=f"{ecr}:already-mirrored")
    with _mock_engine_start(), patch.object(engine.ImageBuilder, "ensure_mirrored") as m:
        await provider.create(spec)
    m.assert_not_called()


async def test_create_skips_mirror_when_auto_mirror_disabled():
    ecr = "123.dkr.ecr.us-east-1.amazonaws.com/mirror"
    provider = create_provider(_provider_config(ecr_repository=ecr, auto_mirror=False))
    spec = SandboxSpec(image="docker.io/swebench/sweb.eval:latest")
    with _mock_engine_start(), patch.object(engine.ImageBuilder, "ensure_mirrored") as m:
        await provider.create(spec)
    m.assert_not_called()


def test_get_ecr_image_tag_is_content_addressed(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch")
    tag1 = engine.ImageBuilder.get_ecr_image_tag(tmp_path, "env")
    tag2 = engine.ImageBuilder.get_ecr_image_tag(tmp_path, "env")
    assert tag1 == tag2 and tag1.startswith("env__")
    (tmp_path / "Dockerfile").write_text("FROM alpine")
    assert engine.ImageBuilder.get_ecr_image_tag(tmp_path, "env") != tag1


# ── Phase A review fixes ──────────────────────────────────────────────


def test_ephemeral_storage_block_validates_fargate_range():
    # Omit the field (take Fargate's implicit 20 GiB) unless a larger size is requested; an explicit
    # 20 -- or anything outside 21-200 -- is rejected by RegisterTaskDefinition.
    assert engine._ephemeral_storage_block(None) is None
    assert engine._ephemeral_storage_block(0) is None
    assert engine._ephemeral_storage_block(50) == {"sizeInGiB": 50}
    for bad in (20, 201):
        with pytest.raises(ValueError, match="21 and 200"):
            engine._ephemeral_storage_block(bad)


def test_render_env_value_rejects_unresolved_task_ip():
    cfg = engine.EcsFargateConfig(region="us-east-1")
    sandbox = engine.EcsFargateSandbox(engine.SandboxSpec(image="img"), ecs_config=cfg)
    # _task_ip is unknown until the task is running, so it cannot be baked into container env.
    with pytest.raises(ValueError, match="task_ip"):
        sandbox._render_env_value("BASE=http://{task_ip}:8000")
    # once known it resolves; unrelated values pass through untouched.
    sandbox._task_ip = "10.0.0.5"
    assert sandbox._render_env_value("BASE=http://{task_ip}:8000") == "BASE=http://10.0.0.5:8000"
    assert sandbox._render_env_value("PLAIN=1") == "PLAIN=1"


def test_apply_spec_overrides_maps_resources_and_ttl():
    cfg = engine.EcsFargateConfig(region="us-east-1", cpu="256", memory="512")
    spec = SandboxSpec(
        image="img",
        ttl_s=3600,
        ready_timeout_s=120,
        resources={"cpu": 2, "memory_mib": 4096, "disk_gib": 50},
    )
    out = _apply_spec_overrides(cfg, spec)
    assert out.max_task_lifetime_sec == 3600
    assert out.startup_timeout_sec == 120.0
    assert out.cpu == "2048"  # vCPUs -> Fargate CPU units
    assert out.memory == "4096"
    assert out.ephemeral_storage_gib == 50
    # no per-sandbox requests -> config returned unchanged (same object)
    assert _apply_spec_overrides(cfg, SandboxSpec(image="img")) is cfg


def test_apply_spec_overrides_rejects_gpu():
    from nemo_gym.sandbox.providers import SandboxCreateError

    cfg = engine.EcsFargateConfig(region="us-east-1")
    with pytest.raises(SandboxCreateError, match="GPU"):
        _apply_spec_overrides(cfg, SandboxSpec(image="img", resources={"gpu": 1}))


# ── Phase B review fixes ──────────────────────────────────────────────


def test_validate_fargate_cpu_memory():
    engine._validate_fargate_cpu_memory(2048, 8192)  # valid pair
    engine._validate_fargate_cpu_memory(256, 512)  # valid low edge
    with pytest.raises(ValueError, match="cpu must be"):
        engine._validate_fargate_cpu_memory(777, 2048)  # unsupported cpu value
    with pytest.raises(ValueError, match="memory for cpu"):
        # the independent-max bug: cpu=256 cannot pair with 8192 MiB
        engine._validate_fargate_cpu_memory(256, 8192)


def test_get_task_public_ip_waits_for_public_ip_when_enabled():
    # assign_public_ip=ENABLED: must keep retrying for the PublicIp to propagate instead of
    # returning the (VPC-internal, unreachable) private IP on the first poll.
    cfg = engine.EcsFargateConfig(region="us-east-1", cluster="c", assign_public_ip=True)
    sandbox = engine.EcsFargateSandbox(engine.SandboxSpec(image="img"), ecs_config=cfg)
    sandbox._task_arn = "task-arn"
    sandbox._ecs = MagicMock()
    sandbox._ecs.describe_tasks.return_value = {
        "tasks": [
            {
                "attachments": [
                    {"type": "ElasticNetworkInterface", "details": [{"name": "networkInterfaceId", "value": "eni-1"}]}
                ]
            }
        ]
    }
    sandbox._ec2 = MagicMock()
    sandbox._ec2.describe_network_interfaces.side_effect = [
        {"NetworkInterfaces": [{"PrivateIpAddress": "10.0.0.1"}]},  # no PublicIp yet
        {"NetworkInterfaces": [{"Association": {"PublicIp": "1.2.3.4"}, "PrivateIpAddress": "10.0.0.1"}]},
    ]
    with patch(f"{_ENG}.time.sleep"):
        assert sandbox._get_task_public_ip() == "1.2.3.4"


def test_delete_s3_object_best_effort():
    cfg = engine.EcsFargateConfig(region="us-east-1", s3_bucket="bucket")
    fake_s3 = MagicMock()
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_s3
    with patch(f"{_ENG}._require_aws_sdks", return_value=(fake_boto3, MagicMock(), MagicMock())):
        engine._delete_s3_object(cfg, "some/key.zip")
    fake_s3.delete_object.assert_called_once_with(Bucket="bucket", Key="some/key.zip")
    # no bucket configured -> no-op (no client built)
    engine._delete_s3_object(engine.EcsFargateConfig(region="us-east-1"), "k")


# ── Phase C review fixes ──────────────────────────────────────────────


def test_validate_image_ref_blocks_injection():
    # Legit references pass (registry/repo/tag and digests).
    engine._validate_image_ref("docker.io/swebench/sweb.eval.x86_64.astropy_1776-12907:latest")
    engine._validate_image_ref("123.dkr.ecr.us-east-1.amazonaws.com/mirror:tag@sha256:abc123")
    engine._validate_image_ref("ubuntu:24.04")
    # Anything with shell metacharacters (the privileged-build injection surface) is refused.
    for bad in ["ubuntu:24.04; rm -rf /", "img$(whoami)", "img`id`", "a && curl evil", "x\nFROM y", ""]:
        with pytest.raises(ValueError, match="Unsafe image reference"):
            engine._validate_image_ref(bad)


def test_build_efs_volumes_inherits_provider_defaults():
    # A volume opts into EFS (efs=True) without naming a filesystem; it inherits the provider-level
    # efs_filesystem_id / efs_access_point_id (from YAML/SSM).
    cfg = engine.EcsFargateConfig(region="us-east-1", efs_filesystem_id="fs-123", efs_access_point_id="fsap-9")
    spec = engine.SandboxSpec(image="img", volumes=[engine.VolumeMount(container_path="/data", efs=True)])
    sandbox = engine.EcsFargateSandbox(spec, ecs_config=cfg)
    task_volumes, mount_points = sandbox._build_efs_volumes()
    efs = task_volumes[0]["efsVolumeConfiguration"]
    assert efs["fileSystemId"] == "fs-123"
    assert efs["authorizationConfig"]["accessPointId"] == "fsap-9"
    assert mount_points[0]["containerPath"] == "/data"


def test_build_efs_volumes_requires_a_filesystem_id():
    cfg = engine.EcsFargateConfig(region="us-east-1")  # no provider-level EFS default
    spec = engine.SandboxSpec(image="img", volumes=[engine.VolumeMount(container_path="/data", efs=True)])
    sandbox = engine.EcsFargateSandbox(spec, ecs_config=cfg)
    with pytest.raises(ValueError, match="no filesystem id"):
        sandbox._build_efs_volumes()
