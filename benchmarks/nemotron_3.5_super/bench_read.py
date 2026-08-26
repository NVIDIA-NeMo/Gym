from time import time

from pydantic import BaseModel
from tqdm.auto import tqdm

from nemo_gym.config_types import AgentServerRef


class AgentRefModel(BaseModel):
    agent_ref: AgentServerRef


start_time = time()
print("Starting read")
with open(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/jiaqiz/data/gym/all_super_env/tau_pivot/super_row_54_1000_synthetic_tau_all_rollouts_leq_60_passrate.jsonl",
) as file:
    lines = list(tqdm(file, desc="Reading file"))

print("Starting json load")
agent_refs = list(map(AgentRefModel.model_validate_json, lines))
print(f"Finished loading {len(agent_refs)} rows in {time() - start_time:.2f}s")
