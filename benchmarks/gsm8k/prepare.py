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
"""Prepare the GSM8K benchmark (test split).

Mirrors nemo-skills' `nemo_skills/dataset/gsm8k/prepare.py` exactly:
same upstream URL, same hardcoded answer fixes, same "<<...>>" calc
stripping from the reference solution, and same int-cast when the
expected answer is an integer value. The only Gym-side difference is
the `problem` -> `question` rename.
"""

import json
import os
import re
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gsm8k_benchmark.jsonl"

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# Known-bad expected answers in the upstream set that Skills patches. Keep
# byte-identical with nemo-skills' gsm8k fixes dict.
FIXES = {
    """Mr. Finnegan has 3 tanks with a capacity of 7000 gallons, 5000 gallons, and 3000 gallons, respectively. If he fills the first tank up to 3/4 full, the second tank with water up to 4/5 of its capacity, and the third tank up to half of its capacity, how many gallons in total are in the tanks?""": 10750,
    """If you double a number and add 5 to the result, then that's 20 more than half of the original number. What's the original number?""": 10,
    """John has 2 hives of bees.  One of the hives has 1000 bees and produces 500 liters of honey.  The second has 20% fewer bees but each bee produces 40% more honey.  How much honey does he produce?""": 2740,
    """Three blue chips are in a jar which is 10% of the entire chips. If 50% of the chips are white and the rest are green, how many green chips are there?""": 15,
    """Janet filmed a new movie that is 60% longer than her previous 2-hour long movie.  Her previous movie cost $50 per minute to film, and the newest movie cost twice as much per minute to film as the previous movie.  What was the total amount of money required to film Janet's entire newest film?""": 19200,
    """Students at Highridge High earn 2 points for each correct answer during a quiz bowl If a student correctly answers all the questions in a round, the student is awarded an additional 4 point bonus. They played a total of five rounds each consisting of five questions. If James only missed one question, how many points did he get?""": 64,
    """Robert and Teddy are planning to buy snacks for their friends.  Robert orders five boxes of pizza at $10 each box and ten cans of soft drinks at $2 each. Teddy buys six hamburgers at $3 each and an additional ten cans of soft drinks. How much do they spend in all?""": 108,
    """James invests $2000 a week into his bank account.  He had $250,000 in his account when the year started.  At the end of the year, he gets a windfall that is worth 50% more than what he has in his bank account.   How much money does he have?""": 531000,
    """James does chores around the class.  There are 3 bedrooms, 1 living room, and 2 bathrooms to clean.  The bedrooms each take 20 minutes to clean.  The living room takes as long as the 3 bedrooms combined.  The bathroom takes twice as long as the living room.  He also cleans the outside which takes twice as long as cleaning the house.  He splits the chores with his 2 siblings who are just as fast as him.  How long, in hours, does he work?""": 6,
    """During one game, a total of 50 people attended a baseball team’s games. Forty percent and thirty-four percent of the audiences are supporters of the first and second teams, respectively. How many people attended the game did not support either of the teams?""": 13,
    """Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)""": "07:30",
}


def prepare() -> Path:
    """Download GSM8K test split, apply Skills' transforms, write Gym JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    original_file = DATA_DIR / "original_test.jsonl"
    urllib.request.urlretrieve(URL, original_file)

    count = 0
    try:
        with open(original_file, "rt", encoding="utf-8") as fin, open(OUTPUT_FPATH, "w") as fout:
            for line in fin:
                original_entry = json.loads(line)
                new_entry = {}
                new_entry["question"] = original_entry["question"]
                solution, expected_answer = original_entry["answer"].split("####")
                new_entry["expected_answer"] = float(expected_answer.replace(",", ""))
                if int(new_entry["expected_answer"]) == new_entry["expected_answer"]:
                    new_entry["expected_answer"] = int(new_entry["expected_answer"])
                new_entry["reference_solution"] = re.sub(r"<<.*?>>", "", solution)
                if original_entry["question"] in FIXES:
                    new_entry["expected_answer"] = FIXES[original_entry["question"]]
                fout.write(json.dumps(new_entry) + "\n")
                count += 1
    finally:
        os.remove(original_file)

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
