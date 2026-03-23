import subprocess

import ray


ray.init(address="auto")  # connect to cluster


@ray.remote(num_cpus=0, num_gpus=0)  # no GPU required just to query
def get_gpu_info():
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)
        return output
    except Exception as e:
        return str(e)


# Launch one task per node
node_resource_keys = []
for node in ray.nodes():
    node_resource_key = [k for k in node["Resources"].keys() if k.startswith("node:")][0]
    node_resource_keys.append(node_resource_key)
tasks = [
    get_gpu_info.options(resources={node_resource_key: 0.01}).remote() for node_resource_key in node_resource_keys
]

results = ray.get(tasks)

for i, res in enumerate(results):
    print(f"\n===== Node {i} =====\n")
    print(res)
