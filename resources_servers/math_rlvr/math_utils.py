# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Math utility functions vendored from NeMo-RLVR's ``nemo_rl/evals/math_utils.py``.

Only the subset used by the MATH500-style verifier is reproduced here:
``last_boxed_only_string``, ``remove_boxed``, ``is_equiv``, and the
``strip_string`` helpers they depend on. The original module's
``process_docs``/``process_results`` (which depend on the ``datasets`` package)
are intentionally omitted — verification doesn't need them.

String normalization originally from
https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
"""
from __future__ import annotations


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # Defensive: in practice the verifier never sees this case because the
        # caller pre-checks for ``None`` from ``last_boxed_only_string``.
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if s is None:
        return None

    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return None
        return s[len(left):]

    left = "\\boxed{"
    if s[: len(left)] != left:
        return None
    if s[-1] != "}":
        return None

    return s[len(left):-1]


def last_boxed_only_string(string):
    if string is None:
        return None

    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = remove_right_units(string)

    string = string.replace("\\%", "")
    string = string.replace("\\%", "")  # noqa: W605

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")

    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)

    return string
