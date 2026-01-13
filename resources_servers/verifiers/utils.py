# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import logging
from typing import Any

import verifiers as vf

logger = logging.getLogger(__name__)

def load_verifiers_dataset(
    vf_env: vf.Environment,
    n: int = -1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    # TODO: Is there a more standard way in verifiers to get the dataset? check prime? 
    try:
        dataset = vf_env.get_dataset(n=n, seed=seed)
    except ValueError:
        dataset = None
        for attr in ['dataset', 'train_dataset', 'eval_dataset']:
            ds = getattr(vf_env, attr, None)
            if ds is not None:
                dataset = ds
                logger.info(f"Found dataset in vf_env.{attr}")
                break
        if dataset is None:
            raise ValueError("Environment does not have a dataset")
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            dataset = dataset.select(range(min(n, len(dataset))))

    return [
        {
            "prompt": dataset["prompt"][i],
            "example_id": dataset["example_id"][i],
            "task": dataset["task"][i],
            **({"answer": dataset["answer"][i]} if "answer" in dataset.column_names else {}),
            **({"info": dataset["info"][i]} if "info" in dataset.column_names else {}),
        }
        for i in range(len(dataset))
    ]
