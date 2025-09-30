import json


def process_jsonl_file(input_file, output_file):
    """
    Reads a JSONL file, appends ' - think step by step' to the system
    prompt of each entry, and writes the modified data to a new file.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            # Iterate through each line (which is a JSON object) in the input file
            for line in infile:
                # Load the JSON object from the current line
                data = json.loads(line)

                # Access the 'input' array within the 'responses_create_params' field
                messages = data["responses_create_params"]["input"]

                # Find the system message and append the CoT statement
                for message in messages:
                    if message.get("role") == "system":
                        message["content"] += "Think step by step. Enclose thinking in <think> and </think> tags"
                        break  # Found and modified the system message, so we can break the inner loop

                # Write the modified JSON object to the output file
                outfile.write(json.dumps(data) + "\n")

        print(f"Successfully processed {input_file} and created {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from a line in '{input_file}'.")
    except KeyError:
        print("Error: The expected keys ('responses_create_params' or 'input') were not found in a JSON object.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Define the input and output file names
input_filename = "/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin/data/workbench/train.jsonl"
output_filename = "/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin/data/workbench/train_pre_cot.jsonl"

# input_filename = "/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin/data/workbench/validation.jsonl"
# output_filename = "/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin/data/workbench/validation_pre_cot.jsonl"


# Run the script
process_jsonl_file(input_filename, output_filename)
