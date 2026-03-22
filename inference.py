import json
import os
import time

import torch
from arguments import get_args
from vllm import LLM, SamplingParams


# import shortuuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_vllm_model(args):
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16
    model_path = os.path.join(args.model_folder, args.model_name)
    num_nodes = getattr(args, "num_nodes", 1)
    tensor_parallel_size = args.tensor_parallel_size * num_nodes

    print("load model from %s" % model_path)
    print("torch_dtype:", torch_dtype)
    print("tensor_parallel_size:", tensor_parallel_size)
    if num_nodes > 1:
        print("num_nodes:", num_nodes)
        print("distributed_executor_backend: ray")

    kwargs = dict(
        dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=False,
        enforce_eager=False,
        trust_remote_code=True,
        mamba_ssm_cache_dtype="float32",
        # attention_backend="FLASH_ATTN",
    )
    if num_nodes > 1:
        kwargs["distributed_executor_backend"] = "ray"

    model_vllm = LLM(model_path, **kwargs)

    return model_vllm


def keep_gpu_busy(wait_time_minutes=60, target_utilization=0.5):
    """Keep model in GPU memory with moderate GPU utilization (25-50%).

    Args:
        wait_time_minutes: How long to keep the GPU busy
        target_utilization: Target GPU utilization (0.0-1.0), default 0.35 for ~35%
    """
    gpu_rank = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"\n[GPU {gpu_rank}] Inference complete. Keeping model in GPU memory for {wait_time_minutes} minutes...")
    print(f"[GPU {gpu_rank}] This prevents the node from disappearing while other GPUs are still processing.")
    print(f"[GPU {gpu_rank}] Running dummy GPU computation at ~{target_utilization * 100:.0f}% utilization...")

    # Create tensors for useless computation to keep GPU busy
    start_time = time.time()
    end_time = start_time + wait_time_minutes * 60

    # Allocate GPU tensors for matrix multiplication (smaller size for finer control)
    matrix_size = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)

    # Calculate sleep ratio to achieve target utilization
    # If we compute for T seconds and sleep for S seconds, utilization ≈ T/(T+S)
    # So S = T * (1/utilization - 1)
    sleep_ratio = (1.0 / target_utilization) - 1.0

    iteration = 0
    c = None
    while time.time() < end_time:
        # Time the compute batch
        compute_start = time.time()

        # Perform a small batch of matrix multiplications
        for _ in range(20):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        compute_time = time.time() - compute_start

        # Sleep proportionally to achieve target utilization
        sleep_time = compute_time * sleep_ratio
        if sleep_time > 0:
            time.sleep(sleep_time)

        iteration += 1

        # Print progress every 500 iterations (~every few minutes)
        if iteration % 500 == 0:
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            print(
                f"[GPU {gpu_rank}] GPU busy loop: {elapsed / 60:.1f} min elapsed, {remaining / 60:.1f} min remaining"
            )

    # Clean up
    del a, b
    if c is not None:
        del c
    torch.cuda.empty_cache()

    print(f"[GPU {gpu_rank}] Wait time completed. Exiting...")


def add_chatml_template(prompt, detailed_thinking=False):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"

    # if detailed_thinking:
    #     instruction = "<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step. Detailed thinking on.<|im_end|>\n"
    # else:
    #     instruction = "<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step. Detailed thinking off.<|im_end|>\n"

    # return instruction + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    if detailed_thinking:
        return instruction + "<|im_start|>user\n" + prompt + " /think<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        return instruction + "<|im_start|>user\n" + prompt + " /no_think<|im_end|>\n<|im_start|>assistant\n"


def get_prompt_list(args):
    data_list = None

    ## get input data
    if args.eval_dataset == "livecodebench":
        from data.benchmark_nanov3 import preprocess_livecodebench_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    elif args.eval_dataset == "livecodebench_v6":
        from data.benchmark_nanov3 import preprocess_livecodebench_chatml_template

        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_v6_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    else:
        raise ValueError("please input a correct eval_dataset name!")

    print("number of total prompt_list:", len(prompt_list))
    if args.start_idx != -1 and args.end_idx != -1:
        print("getting data from %d to %d" % (args.start_idx, args.end_idx))
        prompt_list = prompt_list[args.start_idx : args.end_idx]
        if qid_list:
            qid_list = qid_list[args.start_idx : args.end_idx]

    print("number of test samples in the dataset:", len(prompt_list))

    if data_list is not None:
        return prompt_list, qid_list, data_list
    else:
        return prompt_list, qid_list


def main():
    args = get_args(add_evaluation=True)
    if args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    ## load model
    model_vllm = load_vllm_model(args)

    ## load test data
    if args.eval_dataset == "arena_hard_v0_1" or args.eval_dataset == "arena_hard_v2":
        prompt_list, qid_list, data_list = get_prompt_list(args)
    else:
        prompt_list, qid_list = get_prompt_list(args)

    ## run inference
    print("args.max_output_len:", args.max_output_len)

    stop_tokens = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>"]
    if args.topp < 1:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.topp,
            max_tokens=args.max_output_len,
            seed=args.seed,
            stop=stop_tokens,
        )
        print("args.seed:", args.seed)
        print("args.topp:", args.topp)
        print("args.temperature:", args.temperature)
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.topk,
            max_tokens=args.max_output_len,
            seed=args.seed,
            stop=stop_tokens,
        )

    output_list = []
    total_tokens = 0
    gen_start_time = time.time()
    num_samples = len(prompt_list)
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size

    for batch_idx, i in enumerate(range(0, num_samples, args.batch_size)):
        batch_prompts = prompt_list[i : i + args.batch_size]
        if qid_list:
            batch_qids = qid_list[i : i + args.batch_size]

        if args.eval_dataset in ("ifeval", "alpaca_eval"):
            batch_prompts = [add_chatml_template(prompt, not args.disable_thinking) for prompt in batch_prompts]

        outputs = model_vllm.generate(batch_prompts, sampling_params)

        total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
        done = len(output_list) + len(outputs)
        elapsed = time.time() - gen_start_time
        print(
            f"[batch {batch_idx + 1}/{num_batches}] {done}/{num_samples} samples, "
            f"{total_tokens} out tokens, {total_tokens / elapsed:.0f} tok/s, {elapsed:.0f}s elapsed"
        )

        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            if "<|im_end|>" in generated_text:
                idx = generated_text.index("<|im_end|>")
                generated_text = generated_text[:idx]
            if "<|end_of_text|>" in generated_text:
                idx = generated_text.index("<|end_of_text|>")
                generated_text = generated_text[:idx]
            if "<|eot_id|>" in generated_text:
                idx = generated_text.index("<|eot_id|>")
                generated_text = generated_text[:idx]

            if qid_list:
                qid = batch_qids[j]
                output_dict = {"task_id": qid, "output": generated_text}
                output_list.append(output_dict)
            else:
                output_dict = {"output": generated_text}
                output_list.append(output_dict)

    total_elapsed = time.time() - gen_start_time
    print(
        f"Generation done: {len(output_list)} samples, {total_tokens} tokens, "
        f"{total_tokens / total_elapsed:.0f} tok/s, {total_elapsed:.0f}s total"
    )

    ## write to output_datapath
    output_token_k = args.max_output_len // 1000
    if args.temperature == 0.6:
        foldername = "outputs_vllm085_{}k_topp{}_seed{}".format(output_token_k, args.topp, args.seed)
    else:
        foldername = "outputs_vllm085_{}k_temp{}_topp{}_seed{}".format(
            output_token_k, args.temperature, args.topp, args.seed
        )

    output_folder = os.path.join(os.path.join(args.model_folder, args.model_name), foldername)
    # output_folder = os.path.join("/lustre/fsw/portfolios/llmservice/users/wdai/sou_results/1900", foldername)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_name = (
        "%s_%dto%d" % (args.eval_dataset, args.start_idx, args.end_idx)
        if args.start_idx != -1 and args.end_idx != -1
        else args.eval_dataset
    )
    # output_name = output_name + ".jsonl" if qid_list else output_name + ".txt"
    if args.disable_thinking:
        output_name = output_name + "_no_thinking"
    else:
        output_name = output_name + "_thinking"

    output_name = output_name + ".jsonl"

    output_datapath = os.path.join(output_folder, output_name)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w", encoding="utf-8") as f:
        for output in output_list:
            if type(output) == dict:
                f.write(json.dumps(output) + "\n")
            else:
                f.write(output + "\n")

    keep_gpu_busy(wait_time_minutes=5)


if __name__ == "__main__":
    main()
