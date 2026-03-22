import copy
import json
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

# from tools.bigcodebench_eval import untrusted_check
# from latex2sympy2 import latex2sympy
# from tools.livecodebench_eval import _temp_run
from tools.code_verifier_utils import run_test
from tqdm import tqdm


def read_text_data(datapath, print_path=True):
    if print_path:
        print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            # data_list.append(line.strip())
            data_list.append(json.loads(line.strip())["output"])

    return data_list


def read_jsonl_data(datapath):
    print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_item = json.loads(line.strip())
            data_list.append(data_item["output"])

    return data_list


def check_coding_correctness(problem_to_check: Optional[dict], timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(problem_to_check, debug, result, metadata_list, timeout):
        try:
            res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception as e:
            # traceback.print_exc(10)
            result.append([-1 for i in range(len(problem_to_check["input_output"]))])
            metadata_list.append(e)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    total_timeout = (timeout + 1) * len(problem_to_check["input_output"]) + 10
    p = multiprocessing.Process(target=_temp_run, args=(problem_to_check, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()

    judge_value = bool(result and np.all(np.array(result[0]) > 0))

    # if judge_value == False:
    #     if result:
    #         print(result[0], metadata_list[0])
    #     else:
    #         print("Time Limit Exceeded.")
    return judge_value


def update_results(result, timeout=12):
    response_entry = {
        "content": result["generation"],
        "correctness": None,
        "reason": None,
    }

    problem_to_check = copy.deepcopy(result)
    curr_res = check_coding_correctness(problem_to_check, timeout=timeout)

    if curr_res:
        response_entry["correctness"] = True
        response_entry["reason"] = ""
    else:
        response_entry["correctness"] = False
        response_entry["reason"] = "Code is incorrect."

    return response_entry


def evaluate_livecodebench_v2(input_datapath, test_datapath):
    print("reading from %s" % input_datapath)
    id2generation = {}
    with open(input_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            id2generation[item["task_id"]] = item["output"]
    print("length of id2generation:", len(id2generation))

    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        test_list = json.load(f)
    print("length of test_list:", len(test_list))

    count_none = 0
    pattern_code = re.compile(r"```python\n(.*?)```", re.DOTALL)

    combined_results = {}
    for data_item in test_list:
        id_ = data_item["question_id"]
        output = id2generation[id_]

        ## temporary fix
        ## for gpt-oss-120B distilled outputs
        # output = output.replace("sys.stdin.buffer", "sys.stdin").replace(".decode()", "")

        if "</think>" not in output:
            count_none += 1
        else:
            summary = output[output.index("</think>") :]
            if len(pattern_code.findall(summary)) == 0:
                count_none += 1

        all_testcases = data_item["private_test_cases"] + json.loads(data_item["public_test_cases"])

        metadata = json.loads(data_item["metadata"])
        if "func_name" in metadata:
            func_name = metadata["func_name"]
        else:
            func_name = ""

        combined_results[id_] = {
            "input_output": all_testcases,
            "starter_code": func_name,
            "question_id": id_,
            "generation": output,
        }

    total_questions = len(combined_results)
    print("length of combined_results:", total_questions)
    print("count_none:", count_none)

    total_correct = 0
    total_finish = 0
    records = []

    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {}
        for idx, (q_id, result) in enumerate(combined_results.items()):
            future_to_task[executor.submit(update_results, result)] = idx

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Generations",
        ):
            idx = future_to_task[future]
            response_entry = future.result()
            total_correct += response_entry["correctness"]
            total_finish += 1
            records.append(response_entry)

    acc = total_correct / total_questions
    print("accuracy:", acc)

    return acc, count_none


def concat_files(inputfile_list, output_datapath):
    output_list = []
    for inputfile in inputfile_list:
        with open(inputfile, "r") as f:
            for line in f:
                output_list.append(json.loads(line))

    with open(output_datapath, "w") as f:
        for item in tqdm(output_list):
            f.write(json.dumps(item))
            f.write("\n")


def main():
    # model_name = "../../megatron-lm/checkpoints/sft_gptoss_v2_1_32nodes_allpurpose_5e-5_32_262144/safetensors-checkpoint-31216"

    # max_seq_len = 32768
    # max_seq_len = 64000
    # max_seq_len = 90000
    max_seq_len = 131072

    # temperature = 0.6
    temperature = 1.0

    # main_folder = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/sft/ckpts"

    seq_len_str = "_%dk" % (max_seq_len // 1000)
    temp_str = "_temp%.1f" % temperature if temperature == 1.0 else ""
    # seq_len_str = ""
    # temp_str = ""

    # model_folder = os.path.join(main_folder, model_name)
    model_folder = str(sys.argv[1])

    test_datafolder = "/lustre/fsw/portfolios/llmservice/users/wdai/data/foundational_qa/test_benchmarks"

    ## bash install_eval.sh

    avg_acc = []

    # seed_list = ["100", "101", "102", "103", "104", "105", "106", "107", "200", "201", "202", "203", "204", "205", "206", "207", "300", "301", "302", "303", "304", "305", "306", "307", "400", "401", "402", "403", "404", "405", "406", "407", "500", "501", "502", "503", "504", "505", "506", "507", "600", "601", "602", "603", "604", "605", "606", "607", "700", "701", "702", "703", "704", "705", "706", "707", "800", "801", "802", "803", "804", "805", "806", "807"]
    # seed_list = [str(int(item)+11) for item in seed_list]
    seed_list = [
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "333",
        "334",
        "335",
        "336",
        "337",
        "338",
        "339",
        "340",
        "444",
        "445",
        "446",
        "447",
        "448",
        "449",
        "450",
        "451",
        "555",
        "556",
        "557",
        "558",
        "559",
        "560",
        "561",
        "562",
        "666",
        "667",
        "668",
        "669",
        "670",
        "671",
        "672",
        "673",
        "777",
        "778",
        "779",
        "780",
        "781",
        "782",
        "783",
        "784",
        "888",
        "889",
        "890",
        "891",
        "892",
        "893",
        "894",
        "895",
    ]

    # livecodebench_v5
    tmp_list = []
    num_noanswer_list = []
    for seed in seed_list[:8]:
        datapath_exist = True
        input_datapath = os.path.join(
            model_folder,
            "outputs_vllm085%s%s_topp0.95_seed%s/livecodebench_thinking.jsonl" % (seq_len_str, temp_str, seed),
        )
        if not os.path.exists(input_datapath):
            ## concate file
            print("concat files ...")
            idx_list = ["0to93", "93to186", "186to279"]
            # idx_list = ["0to70", "70to140", "140to210", "210to279"]
            # idx_list = ["0to35", "35to70", "70to105", "105to140", "140to175", "175to210", "210to245", "245to279"]
            file_list = []
            for idx in idx_list:
                file_path = os.path.join(
                    model_folder,
                    "outputs_vllm085%s%s_topp0.95_seed%s/livecodebench_%s_thinking.jsonl"
                    % (seq_len_str, temp_str, seed, idx),
                )
                # assert os.path.exists(file_path)
                if not os.path.exists(file_path):
                    print("this file do not exist:", file_path)
                    datapath_exist = False
                    break

                file_list.append(file_path)

            if datapath_exist:
                concat_files(file_list, input_datapath)

        if not datapath_exist:
            print("this path do not exist:", input_datapath)
            continue

        test_datapath = os.path.join(test_datafolder, "livecodebench/test_aug2024tojan2025.json")
        print("=" * 80)
        # tmp_acc = evaluate_livecodebench_new(input_datapath, test_datapath)
        tmp_acc, count_none = evaluate_livecodebench_v2(input_datapath, test_datapath)
        num_noanswer_list.append(count_none)
        tmp_list.append(tmp_acc)
    acc = np.mean(tmp_list)
    avg_num_noanswer = np.mean(num_noanswer_list)
    print("-" * 80)
    print(
        "avg acc for livecodebench v5: %.4f (std of mean: %.4f) (runs: %d)"
        % (acc, np.std(tmp_list) / (len(tmp_list) ** 0.5), len(tmp_list))
    )
    print("avg_num_noanswer for livecodebench v5: %.4f (ratio: %.4f)" % (avg_num_noanswer, avg_num_noanswer / 279))
    avg_acc.append(acc)

    # livecodebench_v6
    # tmp_list = []
    # num_noanswer_list = []
    # for seed in seed_list[:8]:
    #     datapath_exist = True
    #     input_datapath = os.path.join(model_folder, "outputs_vllm085%s%s_topp0.95_seed%s/livecodebench_v6_thinking.jsonl" % (seq_len_str, temp_str, seed))
    #     if not os.path.exists(input_datapath):
    #         ## concate file
    #         print("concat files ...")
    #         idx_list = ["0to22", "22to44", "44to66", "66to88", "88to110", "110to132", "132to154", "154to175"]
    #         file_list = []
    #         for idx in idx_list:
    #             file_path = os.path.join(model_folder,
    #                 "outputs_vllm085%s%s_topp0.95_seed%s/livecodebench_v6_%s_thinking.jsonl" % (seq_len_str, temp_str, seed, idx))
    #             # assert os.path.exists(file_path)
    #             if not os.path.exists(file_path):
    #                 datapath_exist = False
    #                 break

    #             file_list.append(file_path)

    #         if datapath_exist:
    #             concat_files(file_list, input_datapath)

    #     if not datapath_exist:
    #         print("this path do not exist:", input_datapath)
    #         continue

    #     test_datapath = os.path.join(test_datafolder, "livecodebench/test_feb2025toApr2025.json")
    #     print("="*80)
    #     # tmp_acc = evaluate_livecodebench_new(input_datapath, test_datapath)
    #     tmp_acc, count_none = evaluate_livecodebench_v2(input_datapath, test_datapath)
    #     num_noanswer_list.append(count_none)
    #     tmp_list.append(tmp_acc)
    # acc = np.mean(tmp_list)
    # avg_num_noanswer = np.mean(num_noanswer_list)
    # print("-"*80)
    # print("avg acc for livecodebench v6: %.4f (std of mean: %.4f) (runs: %d)" % (acc, np.std(tmp_list)/(len(tmp_list)**0.5), len(tmp_list)))
    # print("avg_num_noanswer for livecodebench v6: %.4f (ratio: %.4f)" % (avg_num_noanswer, avg_num_noanswer/175))
    # avg_acc.append(acc)

    # print("="*80)
    # print("average acc across all %d datasets: %.4f" % (len(avg_acc), np.mean(avg_acc)))


if __name__ == "__main__":
    main()
