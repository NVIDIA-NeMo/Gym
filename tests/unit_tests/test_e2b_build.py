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

"""Unit tests for e2b template building (SDK faked; no network)."""

import asyncio
import json
import re
import types

import pytest

from nemo_gym.sandbox.providers.e2b import build as e2b_build
from nemo_gym.sandbox.providers.e2b.build import (
    E2BTemplateBuildError,
    build_template,
    build_templates,
    derive_alias,
    template_exists,
)


class FakeTemplateBuilder:
    def __init__(self, image: str, username=None, password=None) -> None:
        self.image = image
        self.username = username
        self.password = password


class FakeTemplate:
    """Records build calls; ``existing_aliases`` fakes server-side state."""

    builds: list[dict] = []
    existing_aliases: set[str] = set()
    build_error: Exception | None = None
    build_delay_s: float = 0.0
    fail_images: set[str] = set()
    max_concurrent: int = 0
    _in_flight: int = 0

    def from_image(self, image, username=None, password=None):
        return FakeTemplateBuilder(image, username, password)

    @staticmethod
    async def alias_exists(alias, **kwargs):
        return alias in FakeTemplate.existing_aliases

    @staticmethod
    async def build(builder, *, alias=None, cpu_count=None, memory_mb=None, **kwargs):
        FakeTemplate._in_flight += 1
        FakeTemplate.max_concurrent = max(FakeTemplate.max_concurrent, FakeTemplate._in_flight)
        try:
            if FakeTemplate.build_delay_s:
                await asyncio.sleep(FakeTemplate.build_delay_s)
            if FakeTemplate.build_error is not None:
                raise FakeTemplate.build_error
            if builder.image in FakeTemplate.fail_images:
                raise RuntimeError(f"boom: {builder.image}")
            FakeTemplate.builds.append(
                {
                    "image": builder.image,
                    "alias": alias,
                    "cpu_count": cpu_count,
                    "memory_mb": memory_mb,
                    "username": builder.username,
                    "password": builder.password,
                    **kwargs,
                }
            )
            FakeTemplate.existing_aliases.add(alias)
            return types.SimpleNamespace(alias=alias)
        finally:
            FakeTemplate._in_flight -= 1


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTemplate.builds.clear()
    FakeTemplate.existing_aliases.clear()
    FakeTemplate.build_error = None
    FakeTemplate.build_delay_s = 0.0
    FakeTemplate.fail_images = set()
    FakeTemplate.max_concurrent = 0
    FakeTemplate._in_flight = 0
    monkeypatch.setattr(e2b_build, "_require_e2b_sdk", lambda: types.SimpleNamespace(AsyncTemplate=FakeTemplate))


class TestDeriveAlias:
    def test_alias_is_charset_safe_and_traceable(self) -> None:
        alias = derive_alias("ghcr.io/acme/my_task:1.0", 8, 16384)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", alias), "E2B rejects anything outside [A-Za-z0-9_-]"
        assert alias.startswith("my_task-1-0__")

    def test_alias_is_deterministic(self) -> None:
        assert derive_alias("img:1", 2, 1024) == derive_alias("img:1", 2, 1024)

    def test_resources_change_the_alias(self) -> None:
        # E2B bakes cpu/memory into the template, so differing requests must
        # not collide onto one alias.
        assert derive_alias("img:1", 2, 1024) != derive_alias("img:1", 8, 1024)
        assert derive_alias("img:1", 2, 1024) != derive_alias("img:1", 2, 16384)

    def test_digest_disambiguates_same_stem_from_different_repos(self) -> None:
        a = derive_alias("ghcr.io/one/task:1.0", 2, 1024)
        b = derive_alias("ghcr.io/two/task:1.0", 2, 1024)
        assert a.split("__")[0] == b.split("__")[0]
        assert a != b

    def test_handles_digest_references_and_odd_characters(self) -> None:
        alias = derive_alias("registry.io/org/img@sha256:abc123", 2, 1024)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", alias)


class TestBuildTemplate:
    async def test_builds_and_returns_alias(self) -> None:
        alias = await build_template("ghcr.io/acme/task:1.0", cpu_count=8, memory_mb=16384)
        assert alias == derive_alias("ghcr.io/acme/task:1.0", 8, 16384)
        build = FakeTemplate.builds[0]
        assert build["image"] == "ghcr.io/acme/task:1.0"
        assert (build["cpu_count"], build["memory_mb"]) == (8, 16384)

    async def test_explicit_alias_is_used(self) -> None:
        alias = await build_template("ghcr.io/acme/task:1.0", alias="my-alias")
        assert alias == "my-alias"
        assert FakeTemplate.builds[0]["alias"] == "my-alias"

    async def test_existing_template_is_skipped(self) -> None:
        alias = derive_alias("ghcr.io/acme/task:1.0")
        FakeTemplate.existing_aliases.add(alias)
        assert await build_template("ghcr.io/acme/task:1.0") == alias
        assert FakeTemplate.builds == []

    async def test_rebuild_when_skip_existing_false(self) -> None:
        alias = derive_alias("ghcr.io/acme/task:1.0")
        FakeTemplate.existing_aliases.add(alias)
        await build_template("ghcr.io/acme/task:1.0", skip_existing=False)
        assert len(FakeTemplate.builds) == 1

    async def test_registry_credentials_forwarded(self) -> None:
        await build_template("ghcr.io/acme/task:1.0", registry_username="u", registry_password="p")
        assert (FakeTemplate.builds[0]["username"], FakeTemplate.builds[0]["password"]) == ("u", "p")

    async def test_api_params_forwarded(self) -> None:
        await build_template("ghcr.io/acme/task:1.0", api_key="k", api_url="http://gw:8080")
        assert FakeTemplate.builds[0]["api_key"] == "k"
        assert FakeTemplate.builds[0]["api_url"] == "http://gw:8080"

    async def test_failure_is_wrapped(self) -> None:
        FakeTemplate.build_error = RuntimeError("builder offline")
        with pytest.raises(E2BTemplateBuildError, match="builder offline"):
            await build_template("ghcr.io/acme/task:1.0")

    async def test_timeout_is_reported(self) -> None:
        FakeTemplate.build_delay_s = 0.2
        with pytest.raises(E2BTemplateBuildError, match="Timed out"):
            await build_template("ghcr.io/acme/task:1.0", build_timeout_s=0.01)

    async def test_template_exists_is_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(alias, **kwargs):
            raise RuntimeError("gateway down")

        monkeypatch.setattr(FakeTemplate, "alias_exists", boom)
        # A failed existence probe must not abort provisioning.
        assert await template_exists("whatever") is False


class TestBuildTemplates:
    async def test_returns_image_to_alias_mapping(self) -> None:
        images = ["ghcr.io/acme/a:1", "ghcr.io/acme/b:1"]
        mapping = await build_templates(images, cpu_count=4, memory_mb=2048)
        assert set(mapping) == set(images)
        assert mapping["ghcr.io/acme/a:1"] == derive_alias("ghcr.io/acme/a:1", 4, 2048)
        assert len(FakeTemplate.builds) == 2

    async def test_duplicate_images_are_built_once(self) -> None:
        mapping = await build_templates(["img:1", "img:1", "img:1"])
        assert len(mapping) == 1
        assert len(FakeTemplate.builds) == 1

    async def test_concurrency_is_bounded(self) -> None:
        FakeTemplate.build_delay_s = 0.01
        await build_templates([f"img:{i}" for i in range(8)], concurrency=3)
        assert FakeTemplate.max_concurrent <= 3
        assert len(FakeTemplate.builds) == 8

    async def test_failure_aborts_by_default(self) -> None:
        FakeTemplate.fail_images = {"img:2"}
        with pytest.raises(E2BTemplateBuildError):
            await build_templates(["img:1", "img:2"], concurrency=1)

    async def test_continue_on_error_omits_failures(self) -> None:
        # One bad image must not discard a long batch.
        FakeTemplate.fail_images = {"img:2"}
        mapping = await build_templates(["img:1", "img:2", "img:3"], concurrency=1, continue_on_error=True)
        assert set(mapping) == {"img:1", "img:3"}


class TestCli:
    def test_writes_json_mapping(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "map.json"
        rc = e2b_build.main(["--image", "ghcr.io/acme/a:1", "--output", str(out)])
        assert rc == 0
        assert json.loads(out.read_text()) == {"ghcr.io/acme/a:1": derive_alias("ghcr.io/acme/a:1")}

    def test_writes_yaml_template_map(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        out = tmp_path / "map.yaml"
        assert e2b_build.main(["--image", "ghcr.io/acme/a:1", "--output", str(out)]) == 0
        loaded = yaml.safe_load(out.read_text())
        # Shaped so it can be pasted straight under the provider's create block.
        assert list(loaded) == ["template_map"]
        assert loaded["template_map"] == {"ghcr.io/acme/a:1": derive_alias("ghcr.io/acme/a:1")}

    def test_reads_images_file_ignoring_comments(self, tmp_path) -> None:
        images_file = tmp_path / "images.txt"
        images_file.write_text("# comment\nghcr.io/acme/a:1\n\nghcr.io/acme/b:1\n")
        out = tmp_path / "map.json"
        assert e2b_build.main(["--images-file", str(images_file), "--output", str(out)]) == 0
        assert set(json.loads(out.read_text())) == {"ghcr.io/acme/a:1", "ghcr.io/acme/b:1"}

    def test_requires_at_least_one_image(self) -> None:
        with pytest.raises(SystemExit):
            e2b_build.main([])

    def test_nonzero_exit_when_a_build_fails(self, tmp_path) -> None:
        FakeTemplate.fail_images = {"ghcr.io/acme/b:1"}
        out = tmp_path / "map.json"
        rc = e2b_build.main(
            [
                "--image",
                "ghcr.io/acme/a:1",
                "--image",
                "ghcr.io/acme/b:1",
                "--continue-on-error",
                "--concurrency",
                "1",
                "--output",
                str(out),
            ]
        )
        assert rc == 1
        assert set(json.loads(out.read_text())) == {"ghcr.io/acme/a:1"}
