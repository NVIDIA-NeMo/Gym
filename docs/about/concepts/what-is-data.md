(what-is-data)=

# What is Data?
Data is a core component of machine learning, used across training and evaluation. Fundamentally, one data point captures the initial state of the world, and how the state of the world evolved as a consequence of calling an LLM and executing the actions that it took.

Today, a core component of "state" is typically represented by a sequence of OpenAI Chat Completions messages or Responses items, commonly referred to as {term}`a "rollout" or a "trajectory" <Rollout / Trajectory>`. For more complicated tasks, the rollout is typically augmented by some in-memory or database state specific to that task instance in order to comprehensively represent the current state.

Data takes different shapes through the model training process (see {doc}`training-approaches` for more information on training processes).
1. Pre-training: A single datum represents a miniscule-scoped snapshot of the state of the world, and is typically a single document.
2. S

When explaining “What’s in a rollout?”, it would be helpful to describe each task execution record in greater detail. Specifically, clarify what each rollout output represents, where it originates from, and how it is used during training.
Since the focus of the Rollout Collection section is on how we can generate rollouts during RL training, it’s informative to create a section talking about how the different outputs from rollouts can be used in different RL algorithms. For instance, GRPO directly uses the scalar reward value for computing loss, while DPO just focuses on generations that are categorized into good and bad preference pairs based on the reward score. It would also highlight how the different outputs of Rollout Collection can enable different types of RL algorithms.


concept docs should explain what a rollout is

concept docs should explain how rollouts can be used for different training approaches

concept docs should explain how rollouts can be used for evaluation

concept docs should cross link to training tutorials page for related tutorials

Rollouts are the data that are fed into downstream training algorithms like SFT, DPO, or GRPO. Rollouts are also the data that are scored during evaluation.

SFT confusion

Evaluation confusion?


