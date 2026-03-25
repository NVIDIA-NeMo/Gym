# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from argparse import ArgumentParser

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.tavily_search.app import TavilySearchResourcesServer


parser = ArgumentParser()
parser.add_argument("--input_fpath", type=str, required=True)
args = parser.parse_args()


with open(args.input_fpath) as f:
    dicts = list(map(json.loads, f))


class DummySelf:
    class config:
        debug = False


regex_rewards = []
judge_rewards = []
alignments = []
for d in dicts:
    response = NeMoGymResponse.model_validate(d["response"])

    judge_prompt = d["judge_response_create_params"]["input"][0]["content"]
    correct_answer = judge_prompt.split("[correct_answer]: ")[1].split("\n")[0]

    regex_reward = TavilySearchResourcesServer._verify_answer_with_regex(
        DummySelf, ground_truth=correct_answer, response=response.output_text
    ).reward

    judge_reward = d["reward"]

    regex_rewards.append(regex_reward)
    judge_rewards.append(judge_reward)
    alignments.append(regex_reward == judge_reward)

avg_regex_reward = sum(regex_rewards) / len(dicts)
avg_judge_reward = sum(judge_rewards) / len(dicts)
avg_alignment = sum(alignments) / len(dicts)
print(f"""{avg_regex_reward=:.2f} ({sum(regex_rewards)} / {len(dicts)})
{avg_judge_reward=:.2f} ({sum(judge_rewards)} / {len(dicts)})
{avg_alignment=:.2f} ({sum(alignments)} / {len(dicts)})""")
