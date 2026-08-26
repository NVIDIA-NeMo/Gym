# gymv_agent

Multi-turn text-action agent for the Gym-V resources server (`resources_servers/gym_v/`).

Runs a rollout against a Responses-API model server and Gym-V's `/step`
endpoint, extracting the model's action from the **last** `\boxed{...}` token
in the assistant's plain text.
