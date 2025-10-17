import os
import re
from dataclasses import dataclass


@dataclass
class ParallelReasoningUtils:
    # ----------- Genselect ----------- #
    @staticmethod
    def construct_prompt_genselect_reducer(config, original_problem: str, solutions: list[str]) -> str:
        problem = original_problem.replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
        problem = problem.replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
            "",
        )
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "reducer_genselect.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            REDUCER_PROMPT = f.read()

        new_texts = []
        for solution in solutions:
            if "</think>" in solution:
                text = solution.split("</think>")[-1].strip()
                new_texts.append(text)
            else:
                new_texts.append(solution)
        solutions = new_texts

        reducer_prompt = REDUCER_PROMPT.format(
            problem=problem, solutions=solutions, numsolutions=len(solutions), max_idx=(len(solutions) - 1)
        )
        return reducer_prompt.strip()

    @staticmethod
    def parse_genselect_reduction(reducer_output: str) -> str:
        pattern = re.compile(r"Judgment:\s*(\d+)")
        matches = pattern.findall(reducer_output)
        if not matches:
            return None  # no Judgment found
        idx = int(matches[-1])  # take the last one
        return idx
