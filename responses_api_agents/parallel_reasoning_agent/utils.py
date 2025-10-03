import os
import re
from dataclasses import dataclass


THINK_BEGIN_TAG = "<think>"
THINK_END_TAG = "</think>"
PLANNER_BEGIN_TAG = "<plan>"
PLANNER_END_TAG = "</plan>"


@dataclass
class ParallelReasoningUtils:
    # ----------- Planner ----------- #
    @staticmethod
    def construct_prompt_planner_parallelize(original_problem: str, prompt_name: str = None) -> str:
        # Strip dataset boilerplate in the embedded problem to avoid conflicting instructions
        problem = original_problem.replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
        problem = problem.replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
            "",
        )

        if prompt_name is None:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "parallelizer_planner.txt")
        else:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{prompt_name}.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                PLANNER_PROMPT = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Planner prompt file not found at {prompt_path}")

        planner_prompt = PLANNER_PROMPT.format(problem=problem)
        return planner_prompt.strip()

    @staticmethod
    def construct_prompt_planner_execute(original_problem: str, plan: str, prompt_name: str = None) -> str:
        if prompt_name is None:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "executor_planner.txt")
        else:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", f"{prompt_name}.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            EXECUTOR_PROMPT = f.read()
        executor_prompt = EXECUTOR_PROMPT.format(problem=original_problem, plan=plan)
        return executor_prompt.strip()

    @staticmethod
    def parse_plan(planner_output: str) -> str:
        if PLANNER_BEGIN_TAG in planner_output and PLANNER_END_TAG in planner_output:
            if THINK_END_TAG in planner_output:
                processed_text = planner_output.split(THINK_END_TAG)[1]
            else:
                processed_text = planner_output
            plans = re.findall(
                rf"{PLANNER_BEGIN_TAG}\s*(.*?)\s*{PLANNER_END_TAG}",
                processed_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not plans:
                wrapper = re.search(
                    rf"{PLANNER_BEGIN_TAG}(.*?){PLANNER_END_TAG}",
                    processed_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if wrapper:
                    plans = [line.strip() for line in wrapper.group(1).splitlines() if line.strip()]
            plans = [r.strip() for r in plans if r.strip()]
            if len(plans) < 1:
                plans.extend([""] * (1 - len(plans)))
            else:
                plans = plans[:1]
        else:
            plans = [""] * 1
        return plans

    # ----------- Rewriter ----------- #
    @staticmethod
    def construct_prompt_rewriter_parallelize(original_problem: str) -> str:
        # Strip dataset boilerplate in the embedded problem to avoid conflicting instructions
        problem = original_problem.replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
        problem = problem.replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
            "",
        )

        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "parallelizer_rewriter.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                PLANNER_PROMPT = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Rewriter prompt file not found at {prompt_path}")

        planner_prompt = PLANNER_PROMPT.format(problem=problem)
        return planner_prompt.strip()

    @staticmethod
    def construct_prompt_rewriter_execute(original_problem: str, rewrite: str) -> str:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "executor_rewriter.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            EXECUTOR_PROMPT = f.read()
        executor_prompt = EXECUTOR_PROMPT.format(problem=rewrite)
        return executor_prompt.strip()

    @staticmethod
    def parse_rewrite(original_problem: str, planner_output: str, use_identity: bool = False) -> str:
        generated_text = planner_output

        if use_identity:
            problem = original_problem.replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
            problem = problem.replace(
                "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
                "",
            )
            rewrites = [problem]
        else:
            if "<rewrite" in generated_text and "</rewrite" in generated_text:
                import re

                if "</think>" in generated_text:
                    processed_text = generated_text.split("</think>")[1]
                else:
                    processed_text = generated_text

                rewrites = re.findall(
                    r"<rewrite>\s*(.*?)\s*</rewrite>",
                    processed_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if not rewrites:
                    wrapper = re.search(
                        r"<rewrites>(.*?)</rewrites>",
                        processed_text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if wrapper:
                        rewrites = [line.strip() for line in wrapper.group(1).splitlines() if line.strip()]
                rewrites = [r.strip() for r in rewrites if r.strip()]
                if len(rewrites) < 1:
                    rewrites.extend([original_problem] * (1 - len(rewrites)))
                else:
                    rewrites = rewrites[:1]
            else:
                rewrites = [original_problem] * 1

        # TODO(jk): handle multiple rewrites as currently app.py takes only the first rewrite
        return rewrites

    # ----------- Genselect ----------- #
    @staticmethod
    def construct_prompt_genselect_reducer(original_problem: str, solutions: list[str]) -> str:
        problem = original_problem.replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
        problem = problem.replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
            "",
        )
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "reducer_genselect.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            REDUCER_PROMPT = f.read()
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
