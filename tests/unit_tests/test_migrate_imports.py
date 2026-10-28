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
"""`scripts/migrate_imports.py` rewrites old `nemo_gym` paths to the paths that replaced them."""

import importlib.util
import textwrap
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_imports.py"
_spec = importlib.util.spec_from_file_location("migrate_imports", SCRIPT)
migrate_imports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_imports)

MODULES = {
    "nemo_gym.server_utils": "nemo_gym.server_runtime.server_utils",
    "nemo_gym.token_id_capture": "nemo_gym.models.token_capture",
    "nemo_gym.tasks": "nemo_gym.data.tasks",
}
NAMES = {
    "nemo_gym.server_utils": {
        "ServerClient": ("nemo_gym.client.server_client", "ServerClient"),
        "request": ("nemo_gym.client.http_client", "request"),
    },
    "nemo_gym.openai_utils": {"NeMoGymResponse": ("nemo_gym.core.responses_api_types", "NeMoGymResponse")},
}


@pytest.fixture
def migrator():
    return migrate_imports.Migrator(MODULES, NAMES)


def _python(migrator, source: str) -> str:
    rewritten, _ = migrator.rewrite_python(textwrap.dedent(source))
    return rewritten


class TestRewritePython:
    def test_from_import_split_by_destination(self, migrator) -> None:
        source = "from nemo_gym.server_utils import ServerClient, SimpleServer, request as req\n"
        assert _python(migrator, source) == (
            "from nemo_gym.client.server_client import ServerClient\n"
            "from nemo_gym.server_runtime.server_utils import SimpleServer\n"
            "from nemo_gym.client.http_client import request as req\n"
        )

    def test_multiline_from_import_inside_a_function_keeps_indentation(self, migrator) -> None:
        source = """
            def f():
                from nemo_gym.token_id_capture.staging.records import (
                    CallRecord,
                    RolloutReceipt,
                )
                return CallRecord
        """
        assert _python(migrator, source) == textwrap.dedent("""
            def f():
                from nemo_gym.models.token_capture.staging.records import (
                    CallRecord,
                    RolloutReceipt,
                )
                return CallRecord
        """)

    def test_split_inside_a_function_keeps_indentation(self, migrator) -> None:
        source = """
            def f():
                from nemo_gym.server_utils import (
                    ServerClient,
                    SimpleServer,
                )  # comment
        """
        assert _python(migrator, source) == textwrap.dedent("""
            def f():
                from nemo_gym.client.server_client import ServerClient
                from nemo_gym.server_runtime.server_utils import SimpleServer  # comment
        """)

    def test_non_ascii_text_before_an_import_on_the_same_line(self, migrator) -> None:
        source = 'x = "é"; from nemo_gym.server_utils import ServerClient\n'
        assert _python(migrator, source) == 'x = "é"; from nemo_gym.client.server_client import ServerClient\n'

    def test_docs_for_released_versions_are_skipped(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrate_imports, "_load_moves", lambda: (MODULES, NAMES))
        released = tmp_path / "fern" / "versions" / "v0.7.0" / "pages" / "index.mdx"
        current = tmp_path / "fern" / "versions" / "latest" / "pages" / "index.mdx"
        for page in (released, current):
            page.parent.mkdir(parents=True)
            page.write_text("See `nemo_gym.server_utils`.\n")
        assert migrate_imports.main([str(tmp_path / "fern")]) == 0
        assert released.read_text() == "See `nemo_gym.server_utils`.\n"
        assert current.read_text() == "See `nemo_gym.server_runtime.server_utils`.\n"

    def test_docs_redirects_are_skipped(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrate_imports, "_load_moves", lambda: (MODULES, NAMES))
        redirects = tmp_path / "fern" / "docs.yml"
        redirects.parent.mkdir()
        redirects.write_text('  - source: "/nemo/gym/nemo-gym/nemo_gym/server_utils"\n')
        assert migrate_imports.main([str(tmp_path)]) == 0
        assert redirects.read_text() == '  - source: "/nemo/gym/nemo-gym/nemo_gym/server_utils"\n'

    def test_files_that_keep_old_paths_on_purpose_are_skipped(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrate_imports, "_load_moves", lambda: (MODULES, NAMES))
        kept = tmp_path / "nemo_gym" / "_moved_paths.py"
        kept.parent.mkdir()
        kept.write_text('OLD = "nemo_gym.server_utils"\n')
        assert migrate_imports.main([str(tmp_path)]) == 0
        assert kept.read_text() == 'OLD = "nemo_gym.server_utils"\n'

    def test_from_package_import_moved_module(self, migrator) -> None:
        source = "from nemo_gym import server_utils, config_types, PARENT_DIR\n"
        assert _python(migrator, source) == (
            "from nemo_gym.server_runtime import server_utils\nfrom nemo_gym import config_types, PARENT_DIR\n"
        )

    def test_import_statement_and_attribute_chain(self, migrator) -> None:
        source = "import nemo_gym.server_utils as su, os\nimport nemo_gym.tasks.harbor\nnemo_gym.tasks.harbor.run()\n"
        assert _python(migrator, source) == (
            "import nemo_gym.server_runtime.server_utils as su, os\n"
            "import nemo_gym.data.tasks.harbor\n"
            "nemo_gym.data.tasks.harbor.run()\n"
        )

    def test_patch_targets_and_config_strings(self, migrator) -> None:
        source = """
            patch("nemo_gym.server_utils.request")
            patch("nemo_gym.server_utils.get_global_config_dict")
            store = "nemo_gym.token_id_capture.lineage:FileLineageStore"
        """
        assert _python(migrator, source) == textwrap.dedent("""
            patch("nemo_gym.client.http_client.request")
            patch("nemo_gym.server_runtime.server_utils.get_global_config_dict")
            store = "nemo_gym.models.token_capture.lineage:FileLineageStore"
        """)

    def test_similar_names_are_left_alone(self, migrator) -> None:
        source = (
            "import nemo_gym.server_utils_extra\n"
            "x = 'my_nemo_gym.server_utils'\n"
            "from nemo_gym.server_utils_extra import request\n"
            "from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming\n"
        )
        assert _python(migrator, source) == source

    def test_rewriting_twice_changes_nothing_more(self, migrator) -> None:
        source = "from nemo_gym.server_utils import ServerClient, SimpleServer\nimport nemo_gym.tasks\n"
        once = _python(migrator, source)
        assert _python(migrator, once) == once

    def test_import_written_in_a_string_is_rewritten(self, migrator) -> None:
        source = """
            TEMPLATE = \"\"\"
            from nemo_gym.server_utils import SimpleServer, request  # used by the server
            \"\"\"
        """
        assert _python(migrator, source) == textwrap.dedent("""
            TEMPLATE = \"\"\"
            from nemo_gym.server_runtime.server_utils import SimpleServer
            from nemo_gym.client.http_client import request  # used by the server
            \"\"\"
        """)

    def test_patch_of_a_split_out_name_through_its_old_module_is_reported(self, migrator) -> None:
        source = (
            "import nemo_gym.server_utils\n"
            "def test(monkeypatch):\n"
            '    monkeypatch.setattr(nemo_gym.server_utils, "request", None)\n'
            '    monkeypatch.setattr(nemo_gym.server_utils, "SimpleServer", None)\n'
        )
        _, notes = migrator.rewrite_python(source, path="t.py")
        assert len(notes) == 1
        assert notes[0].startswith("t.py:3: patches `request` through `nemo_gym.server_utils`")

    def test_relative_import_into_moved_module_is_reported(self, migrator) -> None:
        _, warnings = migrator.rewrite_python("from . import server_utils\n", path="nemo_gym/x.py")
        assert warnings == ["nemo_gym/x.py:1: relative import may reach a moved module; check by hand"]


class TestRewriteText:
    def test_library_reference_urls_are_left_for_the_docs_redirects(self, migrator) -> None:
        text = "See [ServerClient](/main/nemo-gym/nemo_gym/server_utils) in nemo_gym/server_utils.py.\n"
        assert migrator.rewrite_dotted(text) == (
            "See [ServerClient](/main/nemo-gym/nemo_gym/server_utils) in nemo_gym/server_runtime/server_utils.py.\n"
        )

    def test_code_sample_in_docs(self, migrator) -> None:
        text = "```python\nfrom nemo_gym.server_utils import ServerClient, SESSION_ID_KEY\nfrom nemo_gym.openai_utils import (\n```\n"
        assert migrator.rewrite_dotted(text) == (
            "```python\n"
            "from nemo_gym.client.server_client import ServerClient\n"
            "from nemo_gym.server_runtime.server_utils import SESSION_ID_KEY\n"
            "from nemo_gym.openai_utils import (\n```\n"
        )

    def test_paths_in_codeowners_ci_and_docs(self, migrator) -> None:
        text = (
            "/nemo_gym/server_utils.py @NVIDIA-NeMo/gym_core\n"
            "      - 'nemo_gym/token_id_capture/**'\n"
            "See `nemo_gym.openai_utils.NeMoGymResponse` and nemo_gym/tasks.\n"
            "nemo_gym/tasks_extra/ and nemo_gym/server_utils_extra.py stay.\n"
        )
        assert migrator.rewrite_dotted(text) == (
            "/nemo_gym/server_runtime/server_utils.py @NVIDIA-NeMo/gym_core\n"
            "      - 'nemo_gym/models/token_capture/**'\n"
            "See `nemo_gym.core.responses_api_types.NeMoGymResponse` and nemo_gym/data/tasks.\n"
            "nemo_gym/tasks_extra/ and nemo_gym/server_utils_extra.py stay.\n"
        )


class TestMain:
    def test_rewrites_files_and_check_mode_reports_them(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(migrate_imports, "_load_moves", lambda: (MODULES, NAMES))
        (tmp_path / "app.py").write_text("from nemo_gym.server_utils import ServerClient\n")
        (tmp_path / "README.md").write_text("Import `nemo_gym.server_utils.ServerClient`.\n")
        (tmp_path / "untouched.py").write_text("import os\n")

        assert migrate_imports.main(["--check", str(tmp_path)]) == 1
        assert (tmp_path / "app.py").read_text() == "from nemo_gym.server_utils import ServerClient\n"

        assert migrate_imports.main([str(tmp_path)]) == 0
        assert (tmp_path / "app.py").read_text() == "from nemo_gym.client.server_client import ServerClient\n"
        assert (tmp_path / "README.md").read_text() == "Import `nemo_gym.client.server_client.ServerClient`.\n"
        assert migrate_imports.main(["--check", str(tmp_path)]) == 0

    def test_reads_the_moves_recorded_in_nemo_gym(self) -> None:
        modules, names = migrate_imports._load_moves()
        from nemo_gym._moved_paths import MOVED_MODULES, MOVED_NAMES

        assert set(modules) == set(MOVED_MODULES)
        assert set(names) == set(MOVED_NAMES)
