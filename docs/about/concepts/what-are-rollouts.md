(what-are-rollouts)=

# What are rollouts?
{term}`Rollouts <Rollout / Trajectory>` are 



When explaining “What’s in a rollout?”, it would be helpful to describe each task execution record in greater detail. Specifically, clarify what each rollout output represents, where it originates from, and how it is used during training.
Since the focus of the Rollout Collection section is on how we can generate rollouts during RL training, it’s informative to create a section talking about how the different outputs from rollouts can be used in different RL algorithms. For instance, GRPO directly uses the scalar reward value for computing loss, while DPO just focuses on generations that are categorized into good and bad preference pairs based on the reward score. It would also highlight how the different outputs of Rollout Collection can enable different types of RL algorithms.


concept docs should explain what a rollout is

concept docs should explain how rollouts can be used for different training approaches

concept docs should explain how rollouts can be used for evaluation

concept docs should cross link to training tutorials page for related tutorials

Rollouts are the data that are fed into downstream training algorithms like SFT, DPO, or GRPO. Rollouts are also the data that are scored during evaluation.

SFT confusion

Evaluation confusion?


