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

# ruler's data and init files are generated dynamically based on the provided parameters
# will create multiple subfolders corresponding to different evaluation setups

# From https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/dataset/ruler/prepare.py

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_SETTINGS = """
DATASET_GROUP = "long-context"
METRICS_TYPE = "ruler"
GENERATION_ARGS = (
    "++prompt_config=generic/default "
    "++eval_type=ruler ++eval_config.match_type={match_type} "
{extra_generation_args})
"""

TEXT_COMPLETIONS_EXTRA_ARGS = """\
    "++inference.tokens_to_generate={tokens_to_generate} "
    "++start_assistant_response_key=generation "
    "++inference.endpoint_type=text "
"""


TOKENS_TO_GENERATE = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
MATCH_TYPE = {"niah": "all", "vt": "all", "cwe": "all", "fwe": "all", "qa": "part"}


def prepare_task_for_ns(task, data_dir, setup, data_format):
    """Resaving from data_dir/task/test.jsonl into current folder/task/test.jsonl and adding proper init.py"""
    original_path = Path(data_dir) / task / "test.jsonl"
    new_path = Path(__file__).parent / setup / task / "test.jsonl"
    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    with open(original_path, "r", encoding="utf-8") as fin, open(new_path, "w", encoding="utf-8") as fout:
        for line in fin:
            original_entry = json.loads(line)
            new_entry = {
                "index": original_entry["index"],
                "question": original_entry["input"],
                "expected_answer": original_entry["outputs"],
                "length": original_entry["length"],
            }
            if data_format == "default":
                new_entry["generation"] = original_entry["answer_prefix"].strip()
            elif data_format == "base":
                new_entry["generation"] = "\n" + original_entry["answer_prefix"].strip()
            fout.write(json.dumps(new_entry) + "\n")

    with open(new_path.parent / "__init__.py", "w", encoding="utf-8") as init_file:
        short_name = task.split("_")[0]
        if data_format == "chat":
            extra_generation_args = ""
        else:
            extra_generation_args = TEXT_COMPLETIONS_EXTRA_ARGS.format(
                tokens_to_generate=TOKENS_TO_GENERATE[short_name]
            )

        init_file.write(
            DEFAULT_SETTINGS.format(match_type=MATCH_TYPE[short_name], extra_generation_args=extra_generation_args)
        )


def get_ruler_data(tasks, setup, template_tokens, max_seq_length, data_format, ruler_prepare_args, tmp_data_dir=None):
    if "cwe" in tasks:
        # checking if git-lfs is installed
        try:
            subprocess.run(
                ["git", "lfs", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Git LFS is not installed. Please install it to prepare 'cwe' ruler task")
            exit(1)

    # 1. installing necessary packages
    subprocess.run(["pip install wonderwords html2text tenacity"], check=True, shell=True)

    # 2. use provided tmp_data_dir or create a temporary directory
    if tmp_data_dir is not None:
        tmpdirname = tmp_data_dir
        Path(tmpdirname).mkdir(parents=True, exist_ok=True)
        tmpdir_context = None
    else:
        tmpdir_context = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir_context.__enter__()

    try:
        json_dir = Path(tmpdirname) / "RULER" / "scripts" / "data" / "synthetic" / "json"
        required_files = [
            "english_words.json",
            "hotpotqa.json",
            "PaulGrahamEssays.json",
            "squad.json",
        ]
        # Check if all required files exist
        files_exist = all((json_dir / fname).exists() for fname in required_files)
        if not files_exist:
            subprocess.run(
                "git clone https://github.com/NVIDIA/RULER && "
                "cd RULER/scripts/data/synthetic/json && "
                "python download_paulgraham_essay.py && bash download_qa_dataset.sh",
                check=True,
                shell=True,
                cwd=tmpdirname,
            )

        max_seq_length -= template_tokens  # Adjusting for template tokens

        # preparing the datasets based on user options, in parallel
        def prepare_task(task):
            subprocess.run(
                f"python prepare.py --save_dir {tmpdirname}/ruler_data --benchmark synthetic "
                f"    --subset test --task {task} --tokenizer_type hf --model_template_type base --prepare_for_ns "
                f"    --num_samples 100 --max_seq_length {max_seq_length} {ruler_prepare_args}",
                shell=True,
                check=True,
                cwd=Path(tmpdirname) / "RULER" / "scripts" / "data",
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(prepare_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise exception if any subprocess fails

        # resaving the data and creating __init__.py files
        for task in tasks:
            prepare_task_for_ns(task, Path(tmpdirname) / "ruler_data", setup, data_format=data_format)

        with open(Path(__file__).parent / setup / "__init__.py", "w", encoding="utf-8") as init_file:
            init_file.write("IS_BENCHMARK_GROUP = True\n")
            init_file.write("SCORE_MODULE = 'nemo_skills.dataset.ruler.ruler_score'\n")
            benchmarks = ", ".join(f"'ruler.{setup}.{task}': {{}}" for task in tasks)
            init_file.write(f"BENCHMARKS = {{{benchmarks}}}\n")

    finally:
        if tmpdir_context is not None:
            tmpdir_context.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RULER dataset.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "cwe",
            "fwe",
            "qa_1",
            "qa_2",
        ],
        help="List of tasks to prepare for RULER dataset.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        required=True,
        help="Name of the setup for RULER dataset. Typically should be <model_name>_<sequence_length>.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="Sequence length to check with RULER.",
    )
    parser.add_argument(
        "--template_tokens",
        type=int,
        default=50,
        help="Number of tokens in chat template (will be subtracted from max_seq_length to not exceed max context)",
    )
    parser.add_argument(
        "--tmp_data_dir",
        type=str,
        default=None,
        help="Directory to store intermediate data. If not provided, a temporary directory will be created.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="default",
        choices=["default", "base", "chat"],
        help="""
        default: use default format, answer_prefix is added in the generation field.
        base: use base format, answer_prefix is added in the generation field with a newline separator.
        chat: use chat format, answer_prefix is removed.
        """,
    )

    args, unknown = parser.parse_known_args()
    ruler_prepare_args = " ".join(unknown)
    if not ruler_prepare_args:
        print(
            "ERROR: Can't prepare ruler without arguments provided! "
            "Skipping the preparation step.\n"
            "Example ruler prepare command:\n"
            "ns prepare_data ruler --setup llama_128k "
            "--tokenizer_path meta-llama/Llama-3.1-8B-Instruct --max_seq_length 131072"
        )
        exit(0)

    print(
        f"Preparing RULER dataset for tasks: {args.tasks}, "
        f"data_format: {args.data_format}, "
        f"additional arguments: {ruler_prepare_args}"
    )
    get_ruler_data(
        args.tasks,
        args.setup,
        args.template_tokens,
        args.max_seq_length,
        args.data_format,
        ruler_prepare_args,
        tmp_data_dir=args.tmp_data_dir,
    )
    print("RULER dataset preparation completed.")
