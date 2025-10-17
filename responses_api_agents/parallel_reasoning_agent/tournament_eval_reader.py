import argparse
import json
from collections import defaultdict

import numpy as np


def read_jsonl(args):
    with open(args.results_path) as f:
        data = list(map(json.loads, f))
    return data


def filter_data(data):
    new_data = []
    parallelizer_data = []
    executor_data = []
    reducer_data = []
    for data_row in data:
        for row in data_row["responses"]:
            if row["response"]["metadata"]["stage"] == "final_winner":
                new_data.append(row)
                reducer_data.append(row)
            elif row["response"]["metadata"]["stage"] == "parallelizer":
                executor_data.append(row)
            elif row["response"]["metadata"]["stage"] == "executor":
                executor_data.append(row)
    return {"parallelizer": parallelizer_data, "executor": executor_data, "reducer": reducer_data}


def group_by_problem(data):
    problems = defaultdict(list)
    for row in data:
        problem = row["question"]
        problems[problem].append(row)
    return problems


def pass_at_k(problem_rows):
    rewards = [row["reward"] for row in problem_rows]
    return max(rewards)


def main(args):
    data = read_jsonl(args)
    all_data = filter_data(data)

    final_data = all_data["reducer"]

    rewards = [row["reward"] for row in final_data]
    reward_avg = np.mean(rewards)

    # reducer
    problem_groups = group_by_problem(final_data)
    group_pass_at_k = [pass_at_k(group) for group in problem_groups.values()]
    reward_pass_at_k = np.mean(group_pass_at_k)

    group_size = [len(group) for group in problem_groups.values()]
    group_size = set(group_size)
    assert len(group_size) == 1, len(group_size)
    group_size = group_size.pop()

    print(f"Pass@{group_size}: {reward_pass_at_k:.2%}")
    print(f"Mean@{group_size}: {reward_avg:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--results-path", type=str)
    args = parser.parse_args()
    main(args)
