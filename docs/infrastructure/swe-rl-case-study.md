(swe-rl-case-study)=

# SWE RL Infrastructure Case Study

This case study was written on Feb 13, 2026. The information below may be outdated.


## Background
[SWE RL](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/swe_agents) is a single training environment that is meant to be correlated with agentic coding benchmarks like [SWE Bench Verified](https://openai.com/index/introducing-swe-bench-verified/) and [SWE Bench Multilingual](https://www.swebench.com/multilingual.html). These benchmarks are designed as agentic coding tasks, where the model is provided with a Github repository and an issue from that repository, and the model is asked to resolve that issue.

When people use models for agentic coding purposes, they will provide some prompt (task) and use some harness ([Claude Code](https://code.claude.com/docs/en/overview), [Cursor](https://cursor.com/), [OpenHands](https://openhands.dev/), [Mini SWE Agent](https://github.com/SWE-agent/mini-swe-agent), etc). A harness is just a system prompt provided to the model that instructs it generally what to do, along with the orchestration logic to execute one attempt at the task. The harness is responsible for orchestrating the combination of model and tool calls in order to properly respond back to the user query.

In other words, when models are run on SWE Bench Verified, you actually need to pick a model AND a harness to run it through. This is exactly what's reported in the SWE bench leaderboard https://www.swebench.com/ e.g. "bash only" category which refers to benchmark results across models using a specific harness, or things like "mini-SWE-agent + Claude 4.5 Opus medium (20251101)" which is a harness + a model, respectively.

SWE Bench Verified consists of 500 such tasks that the model needs to accomplish, spread over some number of unique repos. Each repo has its own container that needs to be used, that contains the environment and repo setup for the model.

For training, we use datasets like SWE Gym, R2E gym, etc. Each SWE RL rollout currently can be a sequence of up to 100 alternating model calls and command executions for training, and up to 500 for benchmark.


## Deployment topology during training

When we go to train with NeMo RL, assuming async RL setup, we will provision a set of GPU nodes on the cluster from Slurm and split those into training and generation GPU resources. We will spin NeMo Gym up on the Ray head node which may be colocated with either the train or generation GPUs.

At rollout time, we will make a batch of requests to NeMo Gym which include SWE RL tasks. Inside the SWE RL NeMo Gym server, we will do some config preprocessing, and call ray.remote with SPREAD scheduling policy across our available GPU nodes. Inside the ray.remote call, we will spin up the task instance container using a containerization framework called apptainer. We chose Apptainer (called Singularity at the time we started the effort) because our clusters use enroot as the containerization framework on the Slurm level, and Apptainer was the only containerization framework that we could run from within an enroot container. Looking back, we got lucky this is the choice we made since Apptainer has now pivoted to exactly supporting these agentic coding scenarios.

Inside the apptainer, we will run the harness logic (e.g. OpenHands) to orchestrate calls to the model endpoint and command executions. After the trajectory finishes, we will spin up another instance of the same task container to run final unit tests validation to see whether or not the patch the model generated was correct. If correct, we give reward 1.


## CPU resources necessary
The CPU resources necessary for our SWE RL depend on two primary factors:
1. The tasks themselves
2. The quality of the code that the model generates

At the time of writing this case study, our rough estimation for the tasks we currently have in hand is that each task instance requires around 1 CPU core worth of CPU resources. Preliminary SWE RL experiments typically use a batch size of 16 prompts per step and 32 rollouts per prompt = 512 concurrent task instances. This is roughly 512 CPU cores we need to properly support these experiments. If we run on 16 GPU nodes, that's 32 tasks per node.

However, as task complexity increases over the next few months (especially in more complicated multimodal scenarios), the CPU resources necessary to serve these task instances is only going to increase and at some point we may no longer be able to co-locate on the GPU nodes.

The quality of the code that the model generates is a minor factor. Usually we just need to impose the right resource limits (e.g. max memory, filesystem permissions within the container, etc) so that we don’t crash.


## Current challenges and next steps

Eventually we want to scale SWE RL to match the production RL batch size of up to 4 steps off policy * 256 prompts per step * 16 rollouts per prompt, but our preliminary experiments have shown that 4 steps off policy is too much for the SWE RL setting, and our software (including both NeMo Gym but mainly the model generation speed i.e. vLLM inference software) is not fast enough to support 256x16 so we've kept it at 16x32 for now.

SWE RL is a long horizon task (open research question how to properly train e2e still) and it's also the longest wall clock time training environment by far. For example, the step time for all other RL environments is around 7 minutes, but for SWE the fastest we've been able to go with batch size 512 is around 20 minutes for both train and rollout.
