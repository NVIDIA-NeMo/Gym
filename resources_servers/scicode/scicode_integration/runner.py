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
# ruff: noqa: E501  (embeds helpers ported verbatim from nemo-skills; keep their formatting)

"""Ray executor for SciCode sub-step tests.

A sub-step's accumulated solution is concatenated with the HDF5 target-loading and comparison
helpers (ported verbatim from nemo-skills' scicode_utils.eval_prefix), then the sub-step's test
assertions, and run in a subprocess. Exit code 0 means every assertion passed.

The only change from nemo-skills' eval_prefix is the test-data path: nemo-skills hardcodes
H5PY_FILE = "/data/test_data.h5"; here it is injected from config.
"""

import json
import re
import subprocess
import sys

import ray


# Helper functions ported verbatim from nemo-skills scicode_utils.eval_prefix. H5PY_FILE and the
# h5py/scipy imports are emitted by build_test_program so the test-data path can be parameterized.
_EVAL_HELPERS = """
def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file=H5PY_FILE):
    data_lst = []
    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst


def are_dicts_close(dict1, dict2, atol=1e-8, rtol=1e-5):
    import sympy
    import scipy
    import numpy as np
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (sympy.Symbol, str)):
            if not value1 == value2:
                return False
        elif isinstance(value1, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            value1 = value1.toarray()
            value2 = value2.toarray()
            if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                return False
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if not value1 == value2:
                    return False
    return True


def process_symbol_in_dict(dict):
    import sympy
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(matrix1, matrix2):
    import numpy as np
    dense1 = matrix1.toarray()
    dense2 = matrix2.toarray()
    return np.allclose(dense1, dense2)


def cmp_tuple_or_list(var1, var2):
    import scipy
    import numpy as np
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if not v1 == v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if not v1 == v2:
                    return False
    return True
"""

_STDERR_TAIL = 2000


def sanitize_test(test_case: str) -> str:
    """Drop `from scicode.` / `import scicode` lines; the ported helpers replace them."""
    lines = [
        line for line in test_case.split("\n") if not re.match(r"^\s*(from\s+scicode\.|import\s+scicode)\b", line)
    ]
    return "\n".join(lines)


def build_test_program(full_generation: str, h5_path: str, step_number: str, sanitized_tests: list[str]) -> str:
    """Assemble one sub-step's program: solution + helpers + targets + assertions."""
    header = f"import h5py\nimport scipy\nH5PY_FILE = {json.dumps(h5_path)}\n"
    program = f"{full_generation}\n{header}{_EVAL_HELPERS}\n"
    program += f"targets = process_hdf5_to_tuple('{step_number}', {len(sanitized_tests)})\n"
    for idx, test in enumerate(sanitized_tests):
        program += f"target = targets[{idx}]\n\n{test}\n"
    return program


# Kept as a plain function (then wrapped with ray.remote below) so the executor logic can be
# unit-tested directly without launching Ray.
def run_substep(program: str, timeout_secs: float) -> dict:
    """Run one sub-step program in a subprocess. Exit code 0 == all assertions passed."""
    try:
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True, timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout"}
    passed = proc.returncode == 0
    return {"passed": passed, "error": "" if passed else proc.stderr.decode("utf-8", errors="replace")[-_STDERR_TAIL:]}


run_substep_remote = ray.remote(run_substep)
