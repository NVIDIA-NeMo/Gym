#!/usr/bin/env python3
"""
Script to generate HotPotQA dataset in NeMo Gym format.
Uses the Hugging Face HotPotQA dataset to create examples for NeMo Gym.
"""
import json
import random
import argparse
import os
from datasets import load_dataset
from typing import Dict, Any


def create_nemo_gym_example(question: str, ground_truth: str) -> Dict[str, Any]:
    """Create a NeMo Gym format example for HotPotQA."""
    return {
        "responses_create_params": {
            "input": [
                {
                    "content": "You are a helpful research assistant that answers questions using search tools. Always search for information first, then look up specific details, and finally submit your answer.",
                    "role": "developer"
                },
                {
                    "content": f"Question: {question}",
                    "role": "user"
                }
            ],
            "tools": [
                {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "The entity to search for"
                            }
                        },
                        "required": ["entity"],
                        "additionalProperties": False
                    },
                    "strict": True,
                    "type": "function",
                    "description": "Search for information about an entity"
                },
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "The keyword to look up"
                            }
                        },
                        "required": ["keyword"],
                        "additionalProperties": False
                    },
                    "strict": True,
                    "type": "function",
                    "description": "Look up specific information from the current search results"
                },
                {
                    "name": "submit_answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The final answer"
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    },
                    "strict": True,
                    "type": "function",
                    "description": "Submit the final answer"
                }
            ]
        },
        "ground_truth": ground_truth
    }


def main():
    """Generate HotPotQA examples in NeMo Gym format."""
    parser = argparse.ArgumentParser(description="Generate HotPotQA dataset for NeMo Gym")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to generate")
    parser.add_argument("--output_file", type=str, default="data/hotpotqa_20_examples.jsonl", help="Output file path")
    args = parser.parse_args()
    
    print("Loading HotPotQA dataset from Hugging Face...")
    
    # Load the validation split (dev set) - it has answers unlike test split
    # Remove trust_remote_code parameter as it's deprecated
    dataset = load_dataset("hotpotqa/hotpot_qa", name="fullwiki")
    val_dataset = dataset["validation"]
    
    print(f"Loaded {len(val_dataset)} validation examples")
    
    # Randomly sample examples
    num_examples = min(args.num_examples, len(val_dataset))
    random_indices = random.sample(range(len(val_dataset)), num_examples)
    
    examples = []
    for idx in random_indices:
        item = val_dataset[idx]
        question = item["question"]
        answer = item["answer"]
        
        example = create_nemo_gym_example(question, answer)
        examples.append(example)
        
        print(f"Generated example {len(examples)}: {question[:50]}...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to JSONL file
    with open(args.output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} examples and saved to {args.output_file}")


if __name__ == "__main__":
    main()