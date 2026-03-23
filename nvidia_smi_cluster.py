import subprocess

import ray


ray.init(address="auto")  # connect to cluster


@ray.remote(num_gpus=0)  # no GPU required just to query
def get_gpu_info():
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)
        return output
    except Exception as e:
        return str(e)


# Launch one task per node
nodes = ray.nodes()
tasks = [get_gpu_info.options(resources={f"node:{n['NodeID']}": 0.01}).remote() for n in nodes]

results = ray.get(tasks)

for i, res in enumerate(results):
    print(f"\n===== Node {i} =====\n")
    print(res)
