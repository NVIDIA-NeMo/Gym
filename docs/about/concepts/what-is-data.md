(what-is-data)=

# What is Data?
Data is a core component of machine learning, used across training and evaluation. Fundamentally, one data point captures the initial state of the world, and how the state of the world evolved as a consequence of calling an LLM and executing the actions that it took.

Today, a core component of "state" is typically represented by a sequence of OpenAI Chat Completions messages or Responses items, commonly referred to as {term}`a "rollout" or a "trajectory" <Rollout / Trajectory>`. For more complicated tasks, the rollout is typically augmented by some in-memory or database state specific to that task instance in order to comprehensively represent the current state.

Here is an example from modern-day agentic use cases for models. In this example, the model is acting as a personal assistant, conversing with the user.
The initial state can be represented using Responses create params
```json
{
    "input": [
        {
            "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
            "role": "developer",
            "type": "message"
        },
        {
            "content": "how's the outside?",
            "role": "user",
            "type": "message"
        }
    ],
    "tools": [
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": ""
                    }
                },
                "required": [
                    "city"
                ],
                "additionalProperties": false
            },
            "strict": true,
            "type": "function",
            "description": ""
        }
    ]
}
```




Data takes different shapes through the model training process (see {doc}`training-approaches` for more information on training approaches). Below are some examples of what "state" data corresponds to and how it manifests concretely.
1. Pre-training: A single datum represents a miniscule-scoped snapshot of the state of the world, and is typically a single document. The model is expected to recall the state of the world, exactly as represented in the datum.
2. SFT: A single datum represents . An SFT dataset typically consists of a dataset of 
3. DPO: 


When explaining “What’s in a rollout?”, it would be helpful to describe each task execution record in greater detail. Specifically, clarify what each rollout output represents, where it originates from, and how it is used during training.
Since the focus of the Rollout Collection section is on how we can generate rollouts during RL training, it’s informative to create a section talking about how the different outputs from rollouts can be used in different RL algorithms. For instance, GRPO directly uses the scalar reward value for computing loss, while DPO just focuses on generations that are categorized into good and bad preference pairs based on the reward score. It would also highlight how the different outputs of Rollout Collection can enable different types of RL algorithms.


concept docs should explain what a rollout is

concept docs should explain how rollouts can be used for different training approaches

concept docs should explain how rollouts can be used for evaluation

concept docs should cross link to training tutorials page for related tutorials

Rollouts are the data that are fed into downstream training algorithms like SFT, DPO, or GRPO. Rollouts are also the data that are scored during evaluation.

SFT confusion

Evaluation confusion?


