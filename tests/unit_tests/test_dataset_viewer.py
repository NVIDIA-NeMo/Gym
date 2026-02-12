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
from unittest.mock import mock_open, patch

from pydantic import BaseModel
from pytest import MonkeyPatch

from nemo_gym.dataset_viewer import (
    JsonlDatasetViewerConfig,
    build_jsonl_dataset_viewer,
    get_aggregate_metrics,
)


class TestDatasetViewer:
    def test_sanity(
        self,
    ) -> None:
        config = JsonlDatasetViewerConfig(jsonl_fpath="")

        # With tools
        mock_content = r"""{"reward": 0.0, "accuracy": false, "set_overlap": 0.0, "original_term_minefield_hit": false, "order_instruction_following_failure": false, "id": 44, "expected_synonym_values": [489, 504], "expected_synonyms": ["Awake", "Alert"], "minefield_label": "Alive", "minefield_label_value": 497, "responses_create_params": {"input": [{"content": "# Instructions\nYou are an extraction agent. You will be provided a user query and you need to use the tools provided to you to extract list of synonym values. You will be provided with a bunch of synonyms for each. For each term, please see if it's relevant to the user query and get the values for each synonym as appropriate. You must get and extract the values for every synonym that appears in this list. Please output synonym values in the order they appear in the available synonyms below.\n\n# Available synonyms\nThe term 'Win' has a synonym 'Outperform'.\nThe term 'Empty' has a synonym 'Desolate'.\nThe term 'Open' has a synonym 'Revealed'.\nThe term 'Dry' has a synonym 'Arid'.\nThe term 'Bad' has a synonym 'Wicked'.\nThe term 'Retreat' has a synonym 'Draw back'.\nThe term 'Empty' has a synonym 'Vacant'.\nThe term 'Thick' has a synonym 'Fat'.\nThe term 'Stormy' has a synonym 'Gale-force'.\nThe term 'Rough' has a synonym 'Coarse'.\nThe term 'Day' has a synonym 'Afternoon'.\nThe term 'Quiet' has a synonym 'Hushed'.\nThe term 'Day' has a synonym 'Sunrise'.\nThe term 'Closed' has a synonym 'Covered'.\nThe term 'Early' has a synonym 'Preliminary'.\nThe term 'Night' has a synonym 'Midnight'.\nThe term 'Light' has a synonym 'Clear'.\nThe term 'Wide' has a synonym 'Comprehensive'.\nThe term 'Ugly' has a synonym 'Grotesque'.\nThe term 'Insult' has a synonym 'Belittle'.\nThe term 'Far' has a synonym 'Remote'.\nThe term 'Up' has a synonym 'Higher'.\nThe term 'Stormy' has a synonym 'Tempestuous'.\nThe term 'Dead' has a synonym 'Deceased'.\nThe term 'Dim' has a synonym 'Faint'.\nThe term 'Thick' has a synonym 'Heavy'.\nThe term 'Failure' has a synonym 'Loss'.\nThe term 'Sad' has a synonym 'Depressed'.\nThe term 'Thin' has a synonym 'Slender'.\nThe term 'Dry' has a synonym 'Dehydrated'.\nThe term 'Dirty' has a synonym 'Muddy'.\nThe term 'Fast' has a synonym 'Brisk'.\nThe term 'Defeat' has a synonym 'Failure'.\nThe term 'Sharp' has a synonym 'Tapered'.\nThe term 'Sharp' has a synonym 'Piercing'.\nThe term 'Rough' has a synonym 'Grainy'.\nThe term 'Cowardly' has a synonym 'Craven'.\nThe term 'False' has a synonym 'Incorrect'.\nThe term 'Sad' has a synonym 'Unhappy'.\nThe term 'Brave' has a synonym 'Stouthearted'.\nThe term 'Cowardly' has a synonym 'Yellow'.\nThe term 'Thin' has a synonym 'Skinny'.\nThe term 'Outside' has a synonym 'External'.\nThe term 'Wet' has a synonym 'Soaked'.\nThe term 'Sad' has a synonym 'Heartbroken'.\nThe term 'Success' has a synonym 'Victory'.\nThe term 'Cowardly' has a synonym 'Spineless'.\nThe term 'Tight' has a synonym 'Compact'.\nThe term 'Strong' has a synonym 'Muscular'.\nThe term 'Difficult' has a synonym 'Demanding'.\nThe term 'Old' has a synonym 'Historic'.\nThe term 'Rich' has a synonym 'Opulent'.\nThe term 'Far' has a synonym 'Away'.\nThe term 'Easy' has a synonym 'Effortless'.\nThe term 'Short' has a synonym 'Little'.\nThe term 'Win' has a synonym 'Achieve'.\nThe term 'Compliment' has a synonym 'Extol'.\nThe term 'Advance' has a synonym 'Rise'.\nThe term 'Soft' has a synonym 'Tender'.\nThe term 'Narrow' has a synonym 'Restrictive'.\nThe term 'Dark' has a synonym 'Dusky'.\nThe term 'High' has a synonym 'Prominent'.\nThe term 'Calm' has a synonym 'Undisturbed'.\nThe term 'Closed' has a synonym 'Locked'.\nThe term 'Compulsory' has a synonym 'Statutory'.\nThe term 'Alive' has a synonym 'Awake'.\nThe term 'Weak' has a synonym 'Powerless'.\nThe term 'Difficult' has a synonym 'Grueling'.\nThe term 'Reject' has a synonym 'Rebuff'.\nThe term 'Slow' has a synonym 'Leisurely'.\nThe term 'Clean' has a synonym 'Unsoiled'.\nThe term 'Compulsory' has a synonym 'Obligatory'.\nThe term 'Short' has a synonym 'Diminutive'.\nThe term 'Night' has a synonym 'Nightfall'.\nThe term 'Near' has a synonym 'Proximate'.\nThe term 'Ugly' has a synonym 'Homely'.\nThe term 'Wrong' has a synonym 'Amiss'.\nThe term 'Bad' has a synonym 'Terrible'.\nThe term 'Visible' has a synonym 'Noticeable'.\nThe term 'Near' has a synonym 'In proximity'.\nThe term 'Cold' has a synonym 'Frosty'.\nThe term 'Wrong' has a synonym 'False'.\nThe term 'Soft' has a synonym 'Velvety'.\nThe term 'Day' has a synonym 'Bright'.\nThe term 'Young' has a synonym 'Budding'.\nThe term 'Smelly' has a synonym 'Rancid'.\nThe term 'Low' has a synonym 'Diminished'.\nThe term 'Small' has a synonym 'Microscopic'.\nThe term 'Calm' has a synonym 'Unruffled'.\nThe term 'Empty' has a synonym 'Void'.\nThe term 'Open' has a synonym 'Available'.\nThe term 'Far' has a synonym 'Removed'.\nThe term 'Young' has a synonym 'New'.\nThe term 'Ascend' has a synonym 'Mount'.\nThe term 'Ugly' has a synonym 'Hideous'.\nThe term 'Weak' has a synonym 'Frail'.\nThe term 'Wet' has a synonym 'Damp'.\nThe term 'Tall' has a synonym 'Sky-high'.\nThe term 'Down' has a synonym 'Depressed'.\nThe term 'Happy' has a synonym 'Cheerful'.\nThe term 'Alive' has a synonym 'Alert'.\nThe term 'Easy' has a synonym 'Light'.\nThe term 'Accept' has a synonym 'Receive'.\nThe term 'Advance' has a synonym 'Headway'.\nThe term 'Dim' has a synonym 'Dull'.\nThe term 'Tall' has a synonym 'Towering'.\nThe term 'Fragrant' has a synonym 'Balmy'.\nThe term 'Happy' has a synonym 'Pleased'.\nThe term 'Down' has a synonym 'Drop'.\nThe term 'Hard' has a synonym 'Rigid'.\nThe term 'Loud' has a synonym 'Noisy'.\nThe term 'Light' has a synonym 'Shiny'.\nThe term 'Early' has a synonym 'Prior'.\nThe term 'Hot' has a synonym 'Blazing'.\nThe term 'Light (weight)' has a synonym 'Slim'.\nThe term 'Accept' has a synonym 'Acknowledge'.\nThe term 'Quiet' has a synonym 'Peaceful'.\nThe term 'Outside' has a synonym 'Outdoors'.\nThe term 'Easy' has a synonym 'Painless'.\nThe term 'Success' has a synonym 'Conquest'.\nThe term 'Hard' has a synonym 'Solid'.\nThe term 'Failure' has a synonym 'Setback'.\nThe term 'Low' has a synonym 'Short'.\nThe term 'Late' has a synonym 'Overdue'.\nThe term 'Wet' has a synonym 'Waterlogged'.\nThe term 'Strong' has a synonym 'Forceful'.\nThe term 'Hot' has a synonym 'Warm'.\nThe term 'Dark' has a synonym 'Tenebrous'.\nThe term 'Light (weight)' has a synonym 'Flimsy'.\nThe term 'Smelly' has a synonym 'Pungent'.\nThe term 'Soft' has a synonym 'Mild'.\nThe term 'Early' has a synonym 'First'.\nThe term 'Dirty' has a synonym 'Squalid'.\nThe term 'Dead' has a synonym 'Lifeless'.\nThe term 'Bitter' has a synonym 'Astringent'.\nThe term 'False' has a synonym 'Fallacious'.\nThe term 'Defeat' has a synonym 'Collapse'.\nThe term 'Loud' has a synonym 'Blaring'.\nThe term 'Dull' has a synonym 'Dim'.\nThe term 'Stormy' has a synonym 'Wild'.\nThe term 'Narrow' has a synonym 'Compressed'.\nThe term 'Rich' has a synonym 'Flush'.\nThe term 'Invisible' has a synonym 'Obscured'.\nThe term 'Slow' has a synonym 'Dragging'.\nThe term 'Young' has a synonym 'Juvenile'.\nThe term 'Bitter' has a synonym 'Caustic'.\nThe term 'Old' has a synonym 'Elderly'.\nThe term 'Slow' has a synonym 'Sluggish'.\nThe term 'Ascend' has a synonym 'Go up'.\nThe term 'Down' has a synonym 'Sink'.\nThe term 'Descend' has a synonym 'Subside'.\nThe term 'Small' has a synonym 'Little'.\nThe term 'High' has a synonym 'Soaring'.\nThe term 'Up' has a synonym 'Climb'.\nThe term 'Calm' has a synonym 'Relaxed'.\nThe term 'Rich' has a synonym 'Well-off'.\nThe term 'Light (weight)' has a synonym 'Breezy'.\nThe term 'Wrong' has a synonym 'Inaccurate'.\nThe term 'Dirty' has a synonym 'Soiled'.\nThe term 'Late' has a synonym 'Unpunctual'.\nThe term 'Quiet' has a synonym 'Low'.\nThe term 'Descend' has a synonym 'Dismount'.\nThe term 'Compliment' has a synonym 'Praise'.\nThe term 'Open' has a synonym 'Unsealed'.\nThe term 'Dull' has a synonym 'Blunt'.\nThe term 'Small' has a synonym 'Minor'.\nThe term 'Retreat' has a synonym 'Escape'.\nThe term 'Fast' has a synonym 'Hasty'.\nThe term 'Invisible' has a synonym 'Secret'.\nThe term 'Success' has a synonym 'Achievement'.\nThe term 'Retreat' has a synonym 'Fall back'.\nThe term 'Cold' has a synonym 'Icy'.\nThe term 'Hard' has a synonym 'Stiff'.\nThe term 'Insult' has a synonym 'Deride'.\nThe term 'Night' has a synonym 'Nocturne'.\nThe term 'Tight' has a synonym 'Firm'.\nThe term 'Accept' has a synonym 'Consent'.\nThe term 'Victory' has a synonym 'Supremacy'.\nThe term 'Old' has a synonym 'Vintage'.\nThe term 'Dry' has a synonym 'Desiccated'.\nThe term 'Narrow' has a synonym 'Pinched'.\nThe term 'Clean' has a synonym 'Sterile'.\nThe term 'Visible' has a synonym 'Perceptible'.\nThe term 'Victory' has a synonym 'Win'.\nThe term 'Advance' has a synonym 'Step up'.\nThe term 'Sharp' has a synonym 'Edged'.\nThe term 'Wide' has a synonym 'Outspread'.\nThe term 'Low' has a synonym 'Flat'.\nThe term 'Closed' has a synonym 'Fastened'.\nThe term 'False' has a synonym 'Untrue'.\nThe term 'Brave' has a synonym 'Bold'.\nThe term 'Reject' has a synonym 'Refuse'.\nThe term 'Fragrant' has a synonym 'Perfumed'.\n\n# Example\nFor example, if the user query is \"I'm very warm\", the term you should focus on is \"hot\". According to the synonyms above, the term \"hot\" has the synonyms \"Blazing\" and \"Warm\", in that order. You need to get synonym values for \"Blazing\" and \"Warm\", let's say those are 5 and 6 respectively, and extract the result of those synonym values i.e. [5, 6] with 5 (blazing) first then 6 (warm) since that is the order they appear in the list of synonyms above.", "role": "system"}, {"content": "How does the human body's response to danger highlight the instinct for survival?", "role": "user"}], "parallel_tool_calls": false, "tools": [{"name": "get_synonym_value", "parameters": {"properties": {"synonym": {"type": "string", "title": "Synonym", "description": "The synonym to get the value for."}}, "type": "object", "required": ["synonym"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Get the synonym value for a synonym.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"synonym_value\": {\"type\": \"integer\", \"title\": \"Synonym Value\", \"description\": \"The value for this synonym.\"}}, \"type\": \"object\", \"required\": [\"synonym_value\"]}\n"}, {"name": "extract_synonym_values", "parameters": {"properties": {"synonym_values": {"items": {"type": "integer"}, "type": "array", "title": "Synonym Values", "description": "The synonym values corresponding to the term for the user query."}}, "type": "object", "required": ["synonym_values"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Extract the synonym values you retrieved for the term that is relevant to the user query.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"success\": {\"type\": \"boolean\", \"title\": \"Success\", \"description\": \"Success.\"}}, \"type\": \"object\", \"required\": [\"success\"]}\n"}]}, "response": {"id": "resp_689038d64ad081929f6f36d2f2554431063ce0bbdad9d001", "created_at": 1754282198.0, "error": null, "incomplete_details": null, "instructions": null, "metadata": {}, "model": "gpt-4.1-2025-04-14", "object": "response", "output": [{"content": [{"annotations": [], "text": "fake chat message", "type": "output_text"}], "role": "assistant", "type": "message", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"summary": [{"type": "summary_text", "text": "fake reasoning"}], "type": "reasoning", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"arguments": "{\"synonym\":\"Survival\"}", "call_id": "call_pyKbpFtdag6LL6euwpAJ8UEw", "name": "get_synonym_value", "type": "function_call", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"call_id": "call_pyKbpFtdag6LL6euwpAJ8UEw", "output": "{\"synonym_value\": 860}", "type": "function_call_output"}, {"arguments": "{\"synonym_values\":[860]}", "call_id": "call_N4Kr3NJJohoTaxSL5DkkG4W6", "name": "extract_synonym_values", "type": "function_call", "id": "fc_689038d6c71c8192b7ab0de7cc655d98063ce0bbdad9d001", "status": "completed"}], "parallel_tool_calls": false, "temperature": 1.0, "tool_choice": "auto", "tools": [{"name": "get_synonym_value", "parameters": {"properties": {"synonym": {"type": "string", "title": "Synonym", "description": "The synonym to get the value for."}}, "type": "object", "required": ["synonym"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Get the synonym value for a synonym.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"synonym_value\": {\"type\": \"integer\", \"title\": \"Synonym Value\", \"description\": \"The value for this synonym.\"}}, \"type\": \"object\", \"required\": [\"synonym_value\"]}\n"}, {"name": "extract_synonym_values", "parameters": {"properties": {"synonym_values": {"items": {"type": "integer"}, "type": "array", "title": "Synonym Values", "description": "The synonym values corresponding to the term for the user query."}}, "type": "object", "required": ["synonym_values"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Extract the synonym values you retrieved for the term that is relevant to the user query.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"success\": {\"type\": \"boolean\", \"title\": \"Success\", \"description\": \"Success.\"}}, \"type\": \"object\", \"required\": [\"success\"]}\n"}], "top_p": 1.0, "background": false, "max_output_tokens": null, "max_tool_calls": null, "previous_response_id": null, "prompt": null, "reasoning": {"effort": null, "generate_summary": null, "summary": null}, "service_tier": "default", "status": "completed", "text": {"format": {"type": "text"}}, "top_logprobs": 0, "truncation": "disabled", "usage": {"input_tokens": 2864, "input_tokens_details": {"cached_tokens": 2798}, "output_tokens": 19, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2883}, "user": null, "prompt_cache_key": null, "safety_identifier": null, "store": true}}"""
        with patch("builtins.open", mock_open(read_data=mock_content)):
            build_jsonl_dataset_viewer(config)

        # Without tools in create params. Responses will always have tools
        mock_content = r"""{"reward": 0.0, "accuracy": false, "set_overlap": 0.0, "original_term_minefield_hit": false, "order_instruction_following_failure": false, "id": 44, "expected_synonym_values": [489, 504], "expected_synonyms": ["Awake", "Alert"], "minefield_label": "Alive", "minefield_label_value": 497, "responses_create_params": {"input": [{"content": "# Instructions", "role": "system"}, {"content": "How does the human body's response to danger highlight the instinct for survival?", "role": "user"}]}, "response": {"id": "resp_689038d64ad081929f6f36d2f2554431063ce0bbdad9d001", "created_at": 1754282198.0, "error": null, "incomplete_details": null, "instructions": null, "metadata": {}, "model": "gpt-4.1-2025-04-14", "object": "response", "output": [{"content": [{"annotations": [], "text": "fake chat message", "type": "output_text"}], "role": "assistant", "type": "message", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"summary": [{"type": "summary_text", "text": "fake reasoning"}], "type": "reasoning", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"arguments": "{\"synonym\":\"Survival\"}", "call_id": "call_pyKbpFtdag6LL6euwpAJ8UEw", "name": "get_synonym_value", "type": "function_call", "id": "fc_689038d5d1cc8192b91bbef0069ff82f063ce0bbdad9d001", "status": "completed"}, {"call_id": "call_pyKbpFtdag6LL6euwpAJ8UEw", "output": "{\"synonym_value\": 860}", "type": "function_call_output"}, {"arguments": "{\"synonym_values\":[860]}", "call_id": "call_N4Kr3NJJohoTaxSL5DkkG4W6", "name": "extract_synonym_values", "type": "function_call", "id": "fc_689038d6c71c8192b7ab0de7cc655d98063ce0bbdad9d001", "status": "completed"}], "parallel_tool_calls": false, "temperature": 1.0, "tool_choice": "auto", "tools": [{"name": "get_synonym_value", "parameters": {"properties": {"synonym": {"type": "string", "title": "Synonym", "description": "The synonym to get the value for."}}, "type": "object", "required": ["synonym"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Get the synonym value for a synonym.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"synonym_value\": {\"type\": \"integer\", \"title\": \"Synonym Value\", \"description\": \"The value for this synonym.\"}}, \"type\": \"object\", \"required\": [\"synonym_value\"]}\n"}, {"name": "extract_synonym_values", "parameters": {"properties": {"synonym_values": {"items": {"type": "integer"}, "type": "array", "title": "Synonym Values", "description": "The synonym values corresponding to the term for the user query."}}, "type": "object", "required": ["synonym_values"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Extract the synonym values you retrieved for the term that is relevant to the user query.\nThis operation returns a value that conforms to the following JSON Schema: {\"properties\": {\"success\": {\"type\": \"boolean\", \"title\": \"Success\", \"description\": \"Success.\"}}, \"type\": \"object\", \"required\": [\"success\"]}\n"}], "top_p": 1.0, "background": false, "max_output_tokens": null, "max_tool_calls": null, "previous_response_id": null, "prompt": null, "reasoning": {"effort": null, "generate_summary": null, "summary": null}, "service_tier": "default", "status": "completed", "text": {"format": {"type": "text"}}, "top_logprobs": 0, "truncation": "disabled", "usage": {"input_tokens": 2864, "input_tokens_details": {"cached_tokens": 2798}, "output_tokens": 19, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2883}, "user": null, "prompt_cache_key": null, "safety_identifier": null, "store": true}}"""
        with patch("builtins.open", mock_open(read_data=mock_content)):
            build_jsonl_dataset_viewer(config)

    def test_get_aggregate_metrics(self, monkeypatch: MonkeyPatch):
        class DummySample(BaseModel):
            responses_create_params: dict = {}
            response: dict = {}
            reward: float = 1.0
            accuracy: bool = True
            set_overlap: float = 0.5
            unrelated_list: list = []
            unrelated_dict: dict = {}

        class DummySampleWithStrings(DummySample):
            some_string: str

        samples = [
            DummySample(reward=1.0, accuracy=True, set_overlap=0.5),
            DummySample(reward=0.0, accuracy=False, set_overlap=0.0),
            DummySample(reward=0.5, accuracy=True, set_overlap=1.0),
        ]

        samples_with_strings = [
            DummySampleWithStrings(reward=1.0, accuracy=True, some_string="asdf"),
            DummySampleWithStrings(reward=0.0, accuracy=False, some_string="asdf"),
            DummySampleWithStrings(reward=0.5, accuracy=True, some_string="word1"),
            DummySampleWithStrings(reward=0.5, accuracy=True, some_string="word1"),
            DummySampleWithStrings(reward=0.5, accuracy=True, some_string="word2"),
        ]

        def mock_compute_sample_metrics(line: str):
            metrics = json.loads(line)
            return metrics, False

        monkeypatch.setattr(
            "nemo_gym.train_data_utils.compute_sample_metrics",
            mock_compute_sample_metrics,
        )

        result_1 = get_aggregate_metrics(samples)

        assert "reward" in result_1
        assert "accuracy" in result_1
        assert "set_overlap" in result_1

        assert "unrelated_str" not in result_1
        assert "unrelated_list" not in result_1
        assert "unrelated_dict" not in result_1

        assert "responses_create_params" not in result_1
        assert "response" not in result_1

        # Check computed values
        reward_stats = result_1["reward"]
        assert reward_stats["Total # non-null values"] == 3
        assert reward_stats["Average"] == round(((1.0 + 0.0 + 0.5) / 3), 3)
        assert reward_stats["Min"] == 0.0
        assert reward_stats["Max"] == 1.0

        # Check computed values with bools converted to int
        accuracy_stats = result_1["accuracy"]
        assert accuracy_stats["Total # non-null values"] == 3
        assert accuracy_stats["Average"] == round(((1 + 0 + 1) / 3), 3)
        assert accuracy_stats["Min"] == 0
        assert accuracy_stats["Max"] == 1

        # Check string counts
        result_2 = get_aggregate_metrics(samples_with_strings)

        assert "some_string" in result_2
        assert result_2["some_string"]["unique_count"] == 3
        assert result_2["some_string"]["total_count"] == 5
