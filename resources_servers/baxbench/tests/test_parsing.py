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
"""The answer parser follows upstream's Parser rule for rule."""

from resources_servers.baxbench.parsing import parse_response, strip_reasoning


def test_single_file_takes_the_first_markdown_block():
    text = "intro\n```python\nfirst\n```\nlater\n```python\nsecond\n```"
    assert parse_response(text, is_multi_file=False, code_filename="app.py") == {"app.py": "first"}


def test_single_file_ignores_bash_blocks():
    text = "```bash\npip install flask\n```\n```python\nmain\n```"
    assert parse_response(text, is_multi_file=False, code_filename="app.py") == {"app.py": "main"}


def test_single_file_falls_back_to_code_tags():
    text = "<CODE>\n**package main**\n</CODE>"
    assert parse_response(text, is_multi_file=False, code_filename="main.go") == {"main.go": "package main"}


def test_multi_file_pairs_filepaths_with_blocks():
    text = (
        "<FILEPATH>app/routes.php</FILEPATH>\n```php\n<?php echo 1;\n```\n"
        "<FILEPATH>composer.json</FILEPATH>\n```json\n{}\n```"
    )
    files = parse_response(text, is_multi_file=True, code_filename=None)
    assert files == {"app/routes.php": "<?php echo 1;", "composer.json": "{}"}


def test_multi_file_accepts_heading_paths_and_unwraps_markdown_in_code_tags():
    text = "### config/routes.rb\n<CODE>\n```ruby\nRails.application\n```\n</CODE>"
    assert parse_response(text, is_multi_file=True, code_filename=None) == {"config/routes.rb": "Rails.application"}


def test_multi_file_with_mismatched_counts_is_unparseable():
    text = "<FILEPATH>a.py</FILEPATH>\n<FILEPATH>b.py</FILEPATH>\n```python\nx\n```"
    assert parse_response(text, is_multi_file=True, code_filename=None) is None


def test_no_code_is_unparseable():
    assert parse_response("no code here", is_multi_file=False, code_filename="app.py") is None


def test_paths_outside_the_workdir_are_dropped():
    text = "<FILEPATH>../../etc/passwd</FILEPATH>\n```txt\nx\n```\n<FILEPATH>/abs.txt</FILEPATH>\n```txt\ny\n```"
    assert parse_response(text, is_multi_file=True, code_filename=None) == {}


def test_strip_reasoning_keeps_the_text_after_the_last_think_close():
    assert strip_reasoning("<think>a</think>b</think>final") == "final"
    assert strip_reasoning("plain") == "plain"


def test_a_path_used_as_file_and_directory_is_unparseable():
    text = "<FILEPATH>config</FILEPATH>\n```ruby\nx\n```\n<FILEPATH>config/routes.rb</FILEPATH>\n```ruby\ny\n```"
    assert parse_response(text, is_multi_file=True, code_filename=None) is None


def test_the_workdir_itself_is_not_a_file():
    text = "<FILEPATH>.</FILEPATH>\n```txt\nx\n```\n<FILEPATH>app.py</FILEPATH>\n```python\ny\n```"
    assert parse_response(text, is_multi_file=True, code_filename=None) == {"app.py": "y"}
