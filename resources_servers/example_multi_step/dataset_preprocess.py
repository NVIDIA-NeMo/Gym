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
from copy import deepcopy

from datasets import load_dataset


# TODO: Update to Nv HF Link
ds = load_dataset("Nexusflow/SyntheticExampleMultiStep", split="train", token="")

base_create_params = dict(
    input=[
        {
            "role": "system",
            "content": """# Instructions
You are an extraction agent. You will be provided a user query and you need to use the tools provided to you to extract list of synonym values. You will be provided with a bunch of synonyms for each. For each term, please see if it's relevant to the user query and get the values for each synonym as appropriate. You must get and extract the values for every synonym that appears in this list. Please output synonym values in the order they appear in the available synonyms below.

# Available synonyms
The term 'Win' has a synonym 'Outperform'.
The term 'Empty' has a synonym 'Desolate'.
The term 'Open' has a synonym 'Revealed'.
The term 'Dry' has a synonym 'Arid'.
The term 'Bad' has a synonym 'Wicked'.
The term 'Retreat' has a synonym 'Draw back'.
The term 'Empty' has a synonym 'Vacant'.
The term 'Thick' has a synonym 'Fat'.
The term 'Stormy' has a synonym 'Gale-force'.
The term 'Rough' has a synonym 'Coarse'.
The term 'Day' has a synonym 'Afternoon'.
The term 'Quiet' has a synonym 'Hushed'.
The term 'Day' has a synonym 'Sunrise'.
The term 'Closed' has a synonym 'Covered'.
The term 'Early' has a synonym 'Preliminary'.
The term 'Night' has a synonym 'Midnight'.
The term 'Light' has a synonym 'Clear'.
The term 'Wide' has a synonym 'Comprehensive'.
The term 'Ugly' has a synonym 'Grotesque'.
The term 'Insult' has a synonym 'Belittle'.
The term 'Far' has a synonym 'Remote'.
The term 'Up' has a synonym 'Higher'.
The term 'Stormy' has a synonym 'Tempestuous'.
The term 'Dead' has a synonym 'Deceased'.
The term 'Dim' has a synonym 'Faint'.
The term 'Thick' has a synonym 'Heavy'.
The term 'Failure' has a synonym 'Loss'.
The term 'Sad' has a synonym 'Depressed'.
The term 'Thin' has a synonym 'Slender'.
The term 'Dry' has a synonym 'Dehydrated'.
The term 'Dirty' has a synonym 'Muddy'.
The term 'Fast' has a synonym 'Brisk'.
The term 'Defeat' has a synonym 'Failure'.
The term 'Sharp' has a synonym 'Tapered'.
The term 'Sharp' has a synonym 'Piercing'.
The term 'Rough' has a synonym 'Grainy'.
The term 'Cowardly' has a synonym 'Craven'.
The term 'False' has a synonym 'Incorrect'.
The term 'Sad' has a synonym 'Unhappy'.
The term 'Brave' has a synonym 'Stouthearted'.
The term 'Cowardly' has a synonym 'Yellow'.
The term 'Thin' has a synonym 'Skinny'.
The term 'Outside' has a synonym 'External'.
The term 'Wet' has a synonym 'Soaked'.
The term 'Sad' has a synonym 'Heartbroken'.
The term 'Success' has a synonym 'Victory'.
The term 'Cowardly' has a synonym 'Spineless'.
The term 'Tight' has a synonym 'Compact'.
The term 'Strong' has a synonym 'Muscular'.
The term 'Difficult' has a synonym 'Demanding'.
The term 'Old' has a synonym 'Historic'.
The term 'Rich' has a synonym 'Opulent'.
The term 'Far' has a synonym 'Away'.
The term 'Easy' has a synonym 'Effortless'.
The term 'Short' has a synonym 'Little'.
The term 'Win' has a synonym 'Achieve'.
The term 'Compliment' has a synonym 'Extol'.
The term 'Advance' has a synonym 'Rise'.
The term 'Soft' has a synonym 'Tender'.
The term 'Narrow' has a synonym 'Restrictive'.
The term 'Dark' has a synonym 'Dusky'.
The term 'High' has a synonym 'Prominent'.
The term 'Calm' has a synonym 'Undisturbed'.
The term 'Closed' has a synonym 'Locked'.
The term 'Compulsory' has a synonym 'Statutory'.
The term 'Alive' has a synonym 'Awake'.
The term 'Weak' has a synonym 'Powerless'.
The term 'Difficult' has a synonym 'Grueling'.
The term 'Reject' has a synonym 'Rebuff'.
The term 'Slow' has a synonym 'Leisurely'.
The term 'Clean' has a synonym 'Unsoiled'.
The term 'Compulsory' has a synonym 'Obligatory'.
The term 'Short' has a synonym 'Diminutive'.
The term 'Night' has a synonym 'Nightfall'.
The term 'Near' has a synonym 'Proximate'.
The term 'Ugly' has a synonym 'Homely'.
The term 'Wrong' has a synonym 'Amiss'.
The term 'Bad' has a synonym 'Terrible'.
The term 'Visible' has a synonym 'Noticeable'.
The term 'Near' has a synonym 'In proximity'.
The term 'Cold' has a synonym 'Frosty'.
The term 'Wrong' has a synonym 'False'.
The term 'Soft' has a synonym 'Velvety'.
The term 'Day' has a synonym 'Bright'.
The term 'Young' has a synonym 'Budding'.
The term 'Smelly' has a synonym 'Rancid'.
The term 'Low' has a synonym 'Diminished'.
The term 'Small' has a synonym 'Microscopic'.
The term 'Calm' has a synonym 'Unruffled'.
The term 'Empty' has a synonym 'Void'.
The term 'Open' has a synonym 'Available'.
The term 'Far' has a synonym 'Removed'.
The term 'Young' has a synonym 'New'.
The term 'Ascend' has a synonym 'Mount'.
The term 'Ugly' has a synonym 'Hideous'.
The term 'Weak' has a synonym 'Frail'.
The term 'Wet' has a synonym 'Damp'.
The term 'Tall' has a synonym 'Sky-high'.
The term 'Down' has a synonym 'Depressed'.
The term 'Happy' has a synonym 'Cheerful'.
The term 'Alive' has a synonym 'Alert'.
The term 'Easy' has a synonym 'Light'.
The term 'Accept' has a synonym 'Receive'.
The term 'Advance' has a synonym 'Headway'.
The term 'Dim' has a synonym 'Dull'.
The term 'Tall' has a synonym 'Towering'.
The term 'Fragrant' has a synonym 'Balmy'.
The term 'Happy' has a synonym 'Pleased'.
The term 'Down' has a synonym 'Drop'.
The term 'Hard' has a synonym 'Rigid'.
The term 'Loud' has a synonym 'Noisy'.
The term 'Light' has a synonym 'Shiny'.
The term 'Early' has a synonym 'Prior'.
The term 'Hot' has a synonym 'Blazing'.
The term 'Light (weight)' has a synonym 'Slim'.
The term 'Accept' has a synonym 'Acknowledge'.
The term 'Quiet' has a synonym 'Peaceful'.
The term 'Outside' has a synonym 'Outdoors'.
The term 'Easy' has a synonym 'Painless'.
The term 'Success' has a synonym 'Conquest'.
The term 'Hard' has a synonym 'Solid'.
The term 'Failure' has a synonym 'Setback'.
The term 'Low' has a synonym 'Short'.
The term 'Late' has a synonym 'Overdue'.
The term 'Wet' has a synonym 'Waterlogged'.
The term 'Strong' has a synonym 'Forceful'.
The term 'Hot' has a synonym 'Warm'.
The term 'Dark' has a synonym 'Tenebrous'.
The term 'Light (weight)' has a synonym 'Flimsy'.
The term 'Smelly' has a synonym 'Pungent'.
The term 'Soft' has a synonym 'Mild'.
The term 'Early' has a synonym 'First'.
The term 'Dirty' has a synonym 'Squalid'.
The term 'Dead' has a synonym 'Lifeless'.
The term 'Bitter' has a synonym 'Astringent'.
The term 'False' has a synonym 'Fallacious'.
The term 'Defeat' has a synonym 'Collapse'.
The term 'Loud' has a synonym 'Blaring'.
The term 'Dull' has a synonym 'Dim'.
The term 'Stormy' has a synonym 'Wild'.
The term 'Narrow' has a synonym 'Compressed'.
The term 'Rich' has a synonym 'Flush'.
The term 'Invisible' has a synonym 'Obscured'.
The term 'Slow' has a synonym 'Dragging'.
The term 'Young' has a synonym 'Juvenile'.
The term 'Bitter' has a synonym 'Caustic'.
The term 'Old' has a synonym 'Elderly'.
The term 'Slow' has a synonym 'Sluggish'.
The term 'Ascend' has a synonym 'Go up'.
The term 'Down' has a synonym 'Sink'.
The term 'Descend' has a synonym 'Subside'.
The term 'Small' has a synonym 'Little'.
The term 'High' has a synonym 'Soaring'.
The term 'Up' has a synonym 'Climb'.
The term 'Calm' has a synonym 'Relaxed'.
The term 'Rich' has a synonym 'Well-off'.
The term 'Light (weight)' has a synonym 'Breezy'.
The term 'Wrong' has a synonym 'Inaccurate'.
The term 'Dirty' has a synonym 'Soiled'.
The term 'Late' has a synonym 'Unpunctual'.
The term 'Quiet' has a synonym 'Low'.
The term 'Descend' has a synonym 'Dismount'.
The term 'Compliment' has a synonym 'Praise'.
The term 'Open' has a synonym 'Unsealed'.
The term 'Dull' has a synonym 'Blunt'.
The term 'Small' has a synonym 'Minor'.
The term 'Retreat' has a synonym 'Escape'.
The term 'Fast' has a synonym 'Hasty'.
The term 'Invisible' has a synonym 'Secret'.
The term 'Success' has a synonym 'Achievement'.
The term 'Retreat' has a synonym 'Fall back'.
The term 'Cold' has a synonym 'Icy'.
The term 'Hard' has a synonym 'Stiff'.
The term 'Insult' has a synonym 'Deride'.
The term 'Night' has a synonym 'Nocturne'.
The term 'Tight' has a synonym 'Firm'.
The term 'Accept' has a synonym 'Consent'.
The term 'Victory' has a synonym 'Supremacy'.
The term 'Old' has a synonym 'Vintage'.
The term 'Dry' has a synonym 'Desiccated'.
The term 'Narrow' has a synonym 'Pinched'.
The term 'Clean' has a synonym 'Sterile'.
The term 'Visible' has a synonym 'Perceptible'.
The term 'Victory' has a synonym 'Win'.
The term 'Advance' has a synonym 'Step up'.
The term 'Sharp' has a synonym 'Edged'.
The term 'Wide' has a synonym 'Outspread'.
The term 'Low' has a synonym 'Flat'.
The term 'Closed' has a synonym 'Fastened'.
The term 'False' has a synonym 'Untrue'.
The term 'Brave' has a synonym 'Bold'.
The term 'Reject' has a synonym 'Refuse'.
The term 'Fragrant' has a synonym 'Perfumed'.

# Example
For example, if the user query is "I'm very warm", the term you should focus on is "hot". According to the synonyms above, the term "hot" has the synonyms "Blazing" and "Warm", in that order. You need to get synonym values for "Blazing" and "Warm", let's say those are 5 and 6 respectively, and extract the result of those synonym values i.e. [5, 6] with 5 (blazing) first then 6 (warm) since that is the order they appear in the list of synonyms above.""",
        },
    ],
    tools=[
        {
            "type": "function",
            "name": "get_synonym_value",
            "description": 'Get the synonym value for a synonym.\nThis operation returns a value that conforms to the following JSON Schema: {"properties": {"synonym_value": {"type": "integer", "title": "Synonym Value", "description": "The value for this synonym."}}, "type": "object", "required": ["synonym_value"]}\n',
            "parameters": {
                "properties": {
                    "synonym": {
                        "type": "string",
                        "title": "Synonym",
                        "description": "The synonym to get the value for.",
                    }
                },
                "type": "object",
                "required": ["synonym"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "extract_synonym_values",
            "description": 'Extract the synonym values you retrieved for the term that is relevant to the user query.\nThis operation returns a value that conforms to the following JSON Schema: {"properties": {"success": {"type": "boolean", "title": "Success", "description": "Success."}}, "type": "object", "required": ["success"]}\n',
            "parameters": {
                "properties": {
                    "synonym_values": {
                        "items": {"type": "integer"},
                        "type": "array",
                        "title": "Synonym Values",
                        "description": "The synonym values corresponding to the term for the user query.",
                    }
                },
                "type": "object",
                "required": ["synonym_values"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ],
    parallel_tool_calls=False,
)

ds = ds.rename_columns(
    {
        "intended_label": "minefield_label",
        "intended_label_value": "minefield_label_value",
        "ground_truth_labels": "expected_synonyms",
        "ground_truth_values": "expected_synonym_values",
    }
)
columns = ds.column_names
ds = ds.add_column("id", list(range(len(ds))))

columns = [
    "id",
    "create_params",
    "expected_synonyms",
    "expected_synonym_values",
    "minefield_label",
    "minefield_label_value",
]
final_rows = []
for e in ds:
    create_params = deepcopy(base_create_params)
    create_params["input"].append(
        {
            "role": "user",
            "content": e["query"],
        }
    )
    row = {"create_params": create_params, **e}

    row = {k: row[k] for k in columns}
    final_rows.append(row)

with open("data/example_multi_step_benchmark.jsonl", "w") as f:
    for row in final_rows:
        f.write(json.dumps(row) + "\n")
