(adding-a-benchmark)=

# Adding a benchmark to Gym

The most important principle when adding benchmarks into Gym is ensuring the fidelity of the benchmark. As a result, there are additional steps and best practices to adding a benchmark that are required on top of adding just a training environment (although the steps below are still suggested for training environments).

## Reward profiling

In order to ensure the rough software correctness of your benchmark, please baseline against publicly available models including a mixture of open-source and closed-source models. This process is also known as reward profiling.

1. Typically any open-source model, even smaller ones e.g. [Qwen 3 30B A3B Instruct 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), will suffice as long as the resulting performance of running that model on the benchmark is in a meaningful range e.g. 30%+. Otherwise you should run a larger model that attains a meaningful score. If no open model is able to achieve meaningful performance, this may indicate that you have a software bug or indeed your benchmark is quite difficult.  
2. For closed-source models, as of Feb 11, 2026, it’s typically fine to run just OpenAI models (rather than Anthropic or Gemini models) for a variety of reasons.  
   1. Typically OpenAI models are the most affordable and have good robustness over various benchmarks, while the same cannot be currently said for Anthropic or Gemini models.  
   2. These closed-source models should reach at least the same scores as open-source models, if not significantly better. It is rare that open-source models outperform closed-source models. If this is not the case, your benchmark may have a software bug.

As of Feb 11, 2026, an example suite of models to reward profile on your benchmark is: your policy model of interest, GPT 5 Nano, GPT 5, [Qwen 3 30B A3B Instruct 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), and [Qwen 3 30B A3B Thinking2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507). If the 30B-level Qwen models aren’t strong enough, please consider [Qwen 3 235B A22B Instruct 2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) and [Qwen 3 235B A22B Thinking 2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507). If those models are also not enough, please consider things like [Kimi K2 Instruct](http://moonshotai/Kimi-K2-Instruct) or [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7). 

## Best practices

After running on a variety of models, you should do a quick analysis of some of each models’ failure cases. It’s critical to look at the actual results and rollouts produced by your benchmark because it will help catch many software bugs and issues that may lead to unexpectedly low or high scores.

In order to ensure infra robustness, please also run your benchmark on a mixture of instruct and thinking models. We want to make our benchmark code model-agnostic, and leave the model-specific logic to things like Responses API Model servers in Gym. It should be as simple as tweaking the config of the model server you are using to switch between instruct and thinking models and achieve reasonable and expected scores for both types.

For integrating existing benchmarks into Gym, you must first use the original repository to reproduce the publicly reported numbers first and achieve reproduction success there. In case we run into reproduction issues down the line, this helps decouple the possible causes of our issues. Then, you can integrate the existing benchmark into Gym and rerun against those same models and reproduce the same scores again.

Once you have a stable setup to run your benchmark, please run it a few times on the highest scoring open source model to understand the variance. We want to increase the number of repeats of the benchmark (i.e. average @ k, k is the number of repeats) so that the variance is less than 1%.
