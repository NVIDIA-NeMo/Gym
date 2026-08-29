"""
Code taken from OmniSQL repository which is released under Apache 2.0 License.

https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/process_dataset.sh

License: https://github.com/RUCKBReasoning/OmniSQL/issues/25
"""

set -e

# # BIRD (Training set)
# python process_dataset.py --input_data_file ./data/bird/train/train_enhanced_with_cot.json --output_data_file ./data/train_bird.json --db_path ./data/bird/train/train_databases/ --tables ./data/bird/train/train_tables.json --source bird --mode train --value_limit_num 2 --db_content_index_path ./data/bird/train/db_contents_index

# BIRD (dev)
python build_index.py --input_data_file ./data/bird/dev_20240627/dev.json --output_data_file ./data/dev_bird.json --db_path ./data/bird/dev_20240627/dev_databases/ --tables ./data/bird/dev_20240627/dev_tables.json --source bird --mode dev --value_limit_num 2 --db_content_index_path ./data/bird/dev_20240627/db_contents_index