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

"""Unit tests for the Luxor image build helper."""

import asyncio
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemo_gym.sandbox.providers.luxor import build as luxor_build
from nemo_gym.sandbox.providers.luxor.build import (
    LuxorImageBuildError,
    build_image,
    build_images,
    derive_name,
    import_script,
)


pytestmark = pytest.mark.sandbox

IMAGE = "docker.io/swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-14141:latest"


class FakeSpec:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> "FakeSpec":
            self.calls.append((name, args, kwargs))
            return self

        return record


def ready_build(ref: str = "ws/gym-image@sha256:abc") -> Any:
    return SimpleNamespace(state="READY", image=SimpleNamespace(digest_ref=ref, ref="gym-image"))


class FakeTemplates:
    def __init__(self) -> None:
        self.existing: list[Any] = []
        self.created: list[dict[str, Any]] = []
        self.built: list[tuple[Any, dict[str, Any]]] = []
        self.build_results: dict[str, Any] = {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def list(self, **kwargs: Any) -> list[Any]:
        assert kwargs == {"tags": ["nemo-gym"]}
        return self.existing

    async def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(name=kwargs["name"], latest_build=None)

    async def build(self, template: Any, **kwargs: Any) -> Any:
        self.built.append((template, kwargs))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        result = self.build_results.get(template.name, ready_build(f"ws/{template.name}@sha256:abc"))
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self) -> None:
        self.templates = FakeTemplates()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(
        luxor_build,
        "_require_luxor_sdk",
        lambda: SimpleNamespace(TemplateSpec=FakeSpec, AsyncClient=lambda: fake),
    )
    return fake


def test_derive_name_is_deterministic_and_valid() -> None:
    name = derive_name(IMAGE)
    assert name == derive_name(IMAGE)
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name)
    assert name != derive_name(IMAGE, ["/testbed"])
    assert name != derive_name(IMAGE.replace("14141", "14142"))
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,63}", derive_name("REGISTRY/" + "A_b." * 40 + "@sha256:abc"))


@pytest.mark.parametrize(
    ("image", "paths"),
    [
        ("", ["/a"]),
        ("has space", ["/a"]),
        (IMAGE, []),
        (IMAGE, "/a"),
        (IMAGE, ["a"]),
        (IMAGE, ["/"]),
        (IMAGE, ["/a/../b"]),
    ],
)
def test_invalid_image_or_paths_are_rejected(image: str, paths: Any) -> None:
    with pytest.raises(ValueError):
        derive_name(image, paths)


def test_import_script_is_valid_shell_and_quotes_inputs() -> None:
    script = import_script("docker.io/acme/task:1;rm", ["/testbed", "/opt/my env"])
    assert "'docker.io/acme/task:1;rm'" in script
    assert "'/opt/my env'" in script
    assert "sha256sum -c" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0


@pytest.mark.asyncio
async def test_build_image_creates_and_builds(client: FakeClient) -> None:
    ref = await build_image(IMAGE, client=client, workspace_id="ws-1", build_timeout_s=30)
    name = derive_name(IMAGE)
    assert ref == f"ws/{name}@sha256:abc"
    created = client.templates.created[0]
    assert (created["name"], created["workspace_id"], created["tags"]) == (name, "ws-1", ["nemo-gym"])
    steps = {call[0]: call for call in created["spec"].calls}
    assert steps["from_image"][1] == ("sandbox",)
    assert steps["run"][1] == (import_script(IMAGE),)
    assert steps["resources"][2] == {"cpu_cores": 2, "memory_mb": 8192, "disk_size_gb": 30}
    assert client.templates.built[0][1] == {"wait_for_completion": True, "timeout": 30}


@pytest.mark.asyncio
async def test_build_image_reuses_ready_image_unless_rebuilding(client: FakeClient) -> None:
    existing = SimpleNamespace(name=derive_name(IMAGE), latest_build=ready_build("ws/existing@sha256:1"))
    client.templates.existing = [existing]
    assert await build_image(IMAGE, client=client) == "ws/existing@sha256:1"
    assert client.templates.built == []

    rebuilt = await build_image(IMAGE, client=client, skip_existing=False)
    assert rebuilt == f"ws/{existing.name}@sha256:abc"
    assert client.templates.created == []
    assert client.templates.built[0][0] is existing


@pytest.mark.asyncio
async def test_build_image_rebuilds_template_without_ready_image(client: FakeClient) -> None:
    existing = SimpleNamespace(name=derive_name(IMAGE), latest_build=SimpleNamespace(state="FAILED", image=None))
    client.templates.existing = [existing]
    assert await build_image(IMAGE, client=client) == f"ws/{existing.name}@sha256:abc"
    assert client.templates.created == []


@pytest.mark.asyncio
async def test_build_image_failures_are_wrapped(client: FakeClient) -> None:
    name = derive_name(IMAGE)
    client.templates.build_results[name] = RuntimeError("boom")
    with pytest.raises(LuxorImageBuildError, match="boom"):
        await build_image(IMAGE, client=client)
    client.templates.build_results[name] = asyncio.TimeoutError()
    with pytest.raises(LuxorImageBuildError, match="may still be running"):
        await build_image(IMAGE, client=client)
    client.templates.build_results[name] = SimpleNamespace(state="FAILED", image=None)
    with pytest.raises(LuxorImageBuildError, match="without an image"):
        await build_image(IMAGE, client=client)
    with pytest.raises(ValueError, match="build_timeout_s"):
        await build_image(IMAGE, client=client, build_timeout_s=0)


@pytest.mark.asyncio
async def test_build_images_maps_unique_images_with_bounded_concurrency(client: FakeClient) -> None:
    images = [f"docker.io/acme/task-{index}:1" for index in range(6)]
    mapping = await build_images([*images, images[0]], concurrency=2)
    assert list(mapping) == images
    assert client.templates.max_in_flight <= 2
    assert client.close_calls == 1
    with pytest.raises(ValueError, match="concurrency"):
        await build_images(images, concurrency=0)


@pytest.mark.asyncio
async def test_build_images_failure_handling(client: FakeClient) -> None:
    images = ["docker.io/acme/good:1", "docker.io/acme/bad:1", "docker.io/acme/later:1"]
    client.templates.build_results[derive_name(images[1])] = RuntimeError("boom")

    mapping = await build_images(images, client=client, concurrency=1, continue_on_error=True)
    assert list(mapping) == [images[0], images[2]]
    assert client.close_calls == 0

    client.templates.built.clear()
    with pytest.raises(LuxorImageBuildError):
        await build_images(images, client=client, concurrency=1)
    assert [template.name for template, _ in client.templates.built] == [derive_name(image) for image in images[:2]]


def test_main_writes_image_map(client: FakeClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images_file = tmp_path / "images.txt"
    images_file.write_text("# comment\ndocker.io/acme/a:1\n\ndocker.io/acme/b:1\n")
    output = tmp_path / "image_map.yaml"
    assert luxor_build.main(["--images-file", str(images_file), "--path", "/app", "--output", str(output)]) == 0
    mapping = yaml.safe_load(output.read_text())["image_map"]
    assert mapping == {
        image: f"ws/{derive_name(image, ['/app'])}@sha256:abc"
        for image in ("docker.io/acme/a:1", "docker.io/acme/b:1")
    }

    assert luxor_build.main(["--image", "docker.io/acme/a:1"]) == 0
    assert list(json.loads(capsys.readouterr().out)) == ["docker.io/acme/a:1"]

    client.templates.build_results[derive_name("docker.io/acme/bad:1")] = RuntimeError("boom")
    assert luxor_build.main(["--image", "docker.io/acme/bad:1", "--continue-on-error"]) == 1
    with pytest.raises(SystemExit, match="No images"):
        luxor_build.main([])


def test_missing_sdk_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_luxor_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "luxor_ai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_luxor_import)
    with pytest.raises(ModuleNotFoundError, match=r"nemo-gym\[sandbox\]"):
        luxor_build._require_luxor_sdk()


def test_sdk_template_call_shapes_match_installed_version() -> None:
    import inspect

    luxor = pytest.importorskip("luxor_ai", reason="luxor-ai optional sandbox dependency is not installed")
    spec = (
        luxor.TemplateSpec()
        .from_image("sandbox")
        .workdir("/home/tenki")
        .run("true", workdir="/home/tenki", timeout=1800)
        .resources(cpu_cores=2, memory_mb=8192, disk_size_gb=30)
    )
    spec.validate()
    from tenki_sandbox.aio.resources import AsyncTemplates

    inspect.signature(AsyncTemplates.list).bind(object(), tags=["nemo-gym"])
    inspect.signature(AsyncTemplates.create).bind(object(), workspace_id=None, name="n", spec=spec, tags=["nemo-gym"])
    inspect.signature(AsyncTemplates.build).bind(object(), object(), wait_for_completion=True, timeout=60)
