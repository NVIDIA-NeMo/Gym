# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import io
import json
import os


def unroll_files(input_files, parent_dir: str | None = None):
    if len(input_files) == 0:
        raise ValueError("No files found with the given pattern.")
    total_files = 0
    for file_pattern in input_files:
        if parent_dir is not None:
            file_pattern = os.path.join(parent_dir, file_pattern)
        for file in sorted(glob.glob(file_pattern, recursive=True)):
            total_files += 1
            yield file
    if total_files == 0:
        raise ValueError("No files found with the given pattern.")


def _make_w_io_base(f, mode: str):
    """
    Utility to write a file to disk. If the file is not an IOBase object, it will
    create the directory structure if it doesn't exist and open the file for writing.
    If the file is an IOBase object, it will be returned as is.

    Args:
        f: A file path or an IOBase object.
        mode: Mode for opening the file (default is write mode).
    """
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def _make_r_io_base(f, mode: str):
    """
    Utility to read a file from disk. If the file is not an IOBase object, it will
    read the file using utf-8 encoding by default.
    This prevents some issues in json files with utf-8 characters that dont display properly.

    Args:
        f: A file path or an IOBase object.
        mode: Mode for opening the file (default is read mode).
    """
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f


def jdump(obj, f, mode="w", indent=None, default=str):
    """
    Dump a list of dictionaries to a file in JSONL format.

    Args:
        obj: A list of dictionaries or a single dictionary to be dumped.
        f: A string path to the location on disk.
        mode: Mode for opening the file (default is write mode).
        indent: Indentation level for pretty-printing JSON (default is None).
        default: A function to handle non-serializable entries (default is str).

    Raises:
        ValueError: If the object type is not supported.
    """
    # Open the file in the specified mode
    f = _make_w_io_base(f, mode)

    # Check if the object is a dictionary
    if isinstance(obj, (dict,)):
        obj = [obj]

    # Check if the object is a list
    if isinstance(obj, (list, tuple)):
        for line in obj:
            json.dump(line, f, indent=indent, default=default)
            f.write("\n")

    # Raise an error if the object type is not supported
    else:
        raise ValueError(f"Expected a single or list of dictionaries, but got {type(obj)}.")

    # Close the file
    f.close()


def jload(filepath, mode="r", verbose=False):
    """
    Safely load a list of dictionaries from a JSONL file.
    Assumes each line in the file is a separate JSON object.
    This function can handle multiple files separated by commas, concatenating the results.
    While loading samples, it ignores any lines that cannot be parsed as valid JSON.

    Args:
        filepath: A string path to the location on disk or a comma-separated list of file paths.
        mode: Mode for opening the file (default is read mode).
        verbose: If True, prints the error message for each line that cannot be parsed.

    Returns:
        A list of dictionaries loaded from the JSONL file(s).
    """
    # Check if the filepath is a string and contains commas
    if "," in filepath:
        f_list = filepath.split(",")
    else:
        f_list = [filepath]

    dataset = []
    # Iterate through each file in the list
    for f in f_list:
        f = _make_r_io_base(f, mode)
        for line_id, line in enumerate(f):
            try:
                data = json.loads(line)
                dataset.append(data)
            except Exception:
                if verbose:
                    print(f"[jload] Error parsing line {line_id} in file {f}: {line}")
                continue

        f.close()
    return dataset


def count_newlines(fname, verbose: bool = False):
    """
    Efficiently count the number of newlines in a file using buffered reading.
    This method is faster than reading the entire file into memory.

    Args:
        fname: A string path to the file.
        verbose: If True, prints the number of lines in the file.

    Returns:
        int: The number of newlines in the file.
    """

    def _make_gen(reader):
        while True:
            b = reader(2**16)
            if not b:
                break
            yield b

    if verbose:
        print("Counting newlines in file :", fname)

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))

    if verbose:
        print("Number of lines in file :", count)
    return count


def calculate_chunk_indices(num_samples: int, num_chunks: int, chunk_id: int):
    """
    Calculate the start and end indices for a chunk of data.

    Args:
        num_samples: Total number of samples in the dataset.
        num_chunks: Number of chunks to split the data into.
        chunk_id: Chunk ID (0-indexed).

    Returns:
        start_idx: Start index of the chunk.
        end_idx: End index of the chunk.
    """
    # Chunk instruction_data if chunk_id and num_chunks are provided as int values
    if chunk_id is not None:
        chunk_id = int(chunk_id)
    if num_chunks is not None:
        num_chunks = int(num_chunks)

    assert chunk_id < num_chunks, "Invalid chunk_id or num_chunks. chunk_id should be in the range [0, num_chunks)."
    assert num_chunks >= 0, "num_chunks should be greater than 0."

    # If num_chunks is 0, return indices of full dataset
    if num_chunks == 1:
        return 0, num_samples

    # Calculate the start and end indices for the chunk
    remainder = num_samples % num_chunks
    base_size = num_samples // num_chunks

    extra = 1 if chunk_id < remainder else 0

    if chunk_id < remainder:
        start_idx = chunk_id * (base_size + 1)
    else:
        start_idx = remainder * (base_size + 1) + (chunk_id - remainder) * base_size

    end_idx = start_idx + base_size + extra

    return start_idx, end_idx


def jload_chunk(filepath, num_chunks: int, chunk_id: int, mode="r", verbose=False):
    """
    Utility function that loads into memory only a chunk of a JSONL file at any given moment.
    This is useful for large files that cannot be loaded into memory all at once.
    This function can handle multiple files separated by commas, concatenating the results.
    Assumes each line in the file is a separate JSON object.
    This function skips any lines that cannot be parsed as valid JSON.

    NOTE: Chunk id is 0-indexed, so the first chunk is chunk_id=0. Num_chunks is the total number of chunks.

    Args:
        filepath: A string path to the location on disk or a comma-separated list of file paths.
        num_chunks: Total number of chunks to split the data into.
        chunk_id: Chunk ID (0-indexed).
        mode: Mode for opening the file (default is read mode).
        verbose: If True, prints the error message for each line that cannot be parsed.

    Returns:
        A list of dictionaries loaded from the specified chunk of the JSONL file(s).
    """
    # Chunk instruction_data if chunk_id and num_chunks are provided and int values
    if chunk_id is not None:
        chunk_id = int(chunk_id)
    if num_chunks is not None:
        num_chunks = int(num_chunks)

    assert chunk_id < num_chunks, "Invalid chunk_id or num_chunks. chunk_id should be in the range [0, num_chunks)."

    if num_chunks == 1:
        # If num_chunks is 1, just load the entire file
        return jload(filepath, mode=mode)

    # First, cheeck the global number of lines in the file
    num_lines = count_newlines(filepath)

    # Calculate the start and end indices for the chunk
    start_idx, end_idx = calculate_chunk_indices(num_lines, num_chunks, chunk_id)

    dataset = []
    f = _make_r_io_base(filepath, mode)
    for idx, line in enumerate(f):
        if idx < start_idx:  # Skip lines before the start index
            continue

        if idx >= end_idx:  # Stop reading lines after the end index
            break

        # Load just the lines in the range of start_idx and end_idx
        try:
            data = json.loads(line)
            dataset.append(data)
        except Exception:
            if verbose:
                print(f"[jload_chunk] Error parsing line {idx} in file {filepath}: {line}")
            continue

    f.close()
    return dataset
