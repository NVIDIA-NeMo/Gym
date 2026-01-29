# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
PATCH_GEN_PROMPT = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
{problem_statement}
</issue>
<code>
{content}
</code>

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. ### followed by the file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Important Instructions:
1. Preserve Indentation: The content string must maintain the exact indentation as required by the original code. Each line of the content should be indented to match the indentation level of the surrounding code to ensure proper functionality. For example, if you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!

2. Correct Format: Ensure that each line of content maintains proper indentation. For instance, if the code block is inside a function or a loop, the new content should align with that structure.

Output format requirement: Please put the *SEARCH/REPLACE* edits in a code block, starting with <solution> and ending with </solution>.
Wrap the *SEARCH/REPLACE* edits in ```python...``` blocks. If you have multiple *SEARCH/REPLACE* edits, use a separate ```python...``` block for each one.
"""

PREMISE_TEST_GEN_PROMPT = """You are an expert Python coder and are given:
- An issue description from a code repository.
- (Optional) Relevant file contents or snippets that may need adjustments.

Your task is to generate a complete test that can be used to both reproduce the issue and check whether the issue is resolved.

The complete test should contain the following:
1. Necessary imports
2. Code to reproduce the issue described in the issue text
- If your test script determines that the issue is NOT YET SOLVED, it should return an exit code of 2. This should happen when running your test on the original codebase (before any edits are applied).
- If your test script determines that the issue is SOLVED, it should return an exit code of 0. This should only happen when running your test on an edited codebase that fixes the issue.
- If your test script crashes or something unexpected happens, it should return an exit code of 1.

Here is an example:

```python
import sys

def test_issue():
    try:
        # Setup: Import necessary modules and initialize test conditions
        import some_module  # Replace with actual module
        from some_module import function_to_test  # Replace with actual function

        # Step 1: Define the input that triggers the issue
        input_data = "some input that causes the bug"  # Replace with actual problematic input

        # Step 2: Compute the actual output
        actual_output = function_to_test(input_data)

        # Step 3: Define the expected correct output
        expected_output = "expected correct result"  # Replace with correct expected output

        # Step 4: Compare results
        if actual_output == expected_output:
            sys.exit(0)  # Issue is fixed
        else:
            print(f"Issue still exists. Actual output: {actual_output} != Expected output: {expected_output}")
            sys.exit(2)  # Issue still exists

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        sys.exit(1)  # Unexpected error occurred

if __name__ == "__main__":
    test_issue()
```

Please ensure the generated test reflects the issue described in the provided issue text.
Since you are writing the test script before the issue is resolved, your test should fail and return an exit code of 2. I will run your script without any modifications, so do not leave any placeholders that I need to fill in.

Output format requirement: Please put the complete test in a code block, starting with <solution> and ending with </solution>.
Wrap the complete test in ```python...``` blocks.
"""

TEST_GEN_PROMPT = """
<issue>
{problem_statement}
</issue>
<relevant_files>
{content}
</relevant_files>
<think>
"""


META_JUDGE_SOLUTION_PREMISE = """You are an expert Python coder and are given:
- An issue description from a code repository.
- Relevant file contents or snippets that may need adjustments.
- Several proposed fixes (labeled A, B, C, etc) provided as git diffs.

<task_description>
Your task is to evaluate the proposed code changes strictly with the context provided here:

1) Make sure you understand each proposed fix:
   - Why might someone propose it?
   - Which part of the issue does it aim to address?

2) Based on the problem statement and the snippet shown, decide which fix correctly addresses the stated issue.

3) Choose exactly one fix that you consider best. Have a bias towards simpler solutions that maintain clear and predictable behavior. New abstractions or APIs should only be introduced if simpler approaches are inadequate.

4) Never, ever, refer to any code that is not present here.

IMPORTANT: Ensure that all analysis and justification are provided first BEFORE making any selection.

You must adhere to **ALL** guidelines specified here.

Output format:
- Please put your reasoning tokens in a separate code block, starting with <think> and ending with </think>
- Output a final tag in the format: <solution>[Label of chosen fix]</solution>
For example, if you choose fix A, you should output:
<solution>A</solution>
</task_description>"""
