#!/usr/bin/env python3
"""
Download and prepare the workplace_assistant dataset from HuggingFace.

This script downloads the nvidia/Nemotron-RL-agent-workplace_assistant dataset
and saves it as train.jsonl and validation.jsonl (using a train/val split).
"""

import json
import os
from datasets import load_dataset

def main():
    # Configuration
    output_dir = "resources_servers/workplace_assistant/data"
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "validation.jsonl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files already exist
    train_exists = os.path.exists(train_path)
    val_exists = os.path.exists(val_path)
    
    if train_exists and val_exists:
        print(f"✓ Both train and validation datasets already exist:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print("\nSkipping download. Delete these files to re-download.")
        return
    
    # Download the dataset from HuggingFace
    print("Downloading workplace_assistant dataset from HuggingFace...")
    print("Repo: nvidia/Nemotron-RL-agent-workplace_assistant")
    dataset = load_dataset("nvidia/Nemotron-RL-agent-workplace_assistant")
    
    print(f"\nDataset info:")
    print(f"  Available splits: {list(dataset.keys())}")
    print(f"  Train split size: {len(dataset['train'])}")
    
    # Split the train dataset into train (90%) and validation (10%)
    train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_data = train_test_split['train']
    val_data = train_test_split['test']
    
    print(f"\nSplitting data:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Save train split
    if not train_exists:
        print(f"\nSaving train split to {train_path}...")
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Saved {len(train_data)} train samples")
    else:
        print(f"✓ Train dataset already exists, skipping: {train_path}")
    
    # Save validation split
    if not val_exists:
        print(f"\nSaving validation split to {val_path}...")
        with open(val_path, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Saved {len(val_data)} validation samples")
    else:
        print(f"✓ Validation dataset already exists, skipping: {val_path}")
    
    print("\n" + "="*60)
    print("Dataset download and preparation complete!")
    print("="*60)
    print(f"Train:      {train_path}")
    print(f"Validation: {val_path}")
    print("\nYou can now use these datasets with NeMo-Gym GRPO training.")

if __name__ == "__main__":
    main()

