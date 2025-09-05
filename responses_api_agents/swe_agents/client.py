"""Client example for SWE-bench wrapper agent.

This script demonstrates how to use the SWE-bench wrapper to solve
a real GitHub issue using GPT-4 or other models.
"""

import json
import asyncio
from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


async def main():
    # Load server client
    server_client = ServerClient.load_from_global_config()
    
    # Example 1: Simple SWE-bench problem
    print("=" * 60)
    print("Example 1: astropy__astropy-12907")
    print("=" * 60)
    
    response = await server_client.post(
        server_name="swe_agents",
        url_path="/v1/responses",
        json=NeMoGymResponseCreateParamsNonStreaming(
            input="""Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
[False, True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, False],
[False, False, False, True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True, True, False, False],
[ True, True, False, False],
[False, False, True, True],
[False, False, True, True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?""",
            metadata={
                "instance_id": "astropy__astropy-12907",
                "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
                "dataset_name": "princeton-nlp/SWE-bench_Verified",
                "split": "test"
            },
            # Model and inference parameters
            model="Qwen3-30B-A3B-Instruct-2507", #"gpt-4.1-2025-04-14",
            temperature=1.0,
            max_output_tokens=32768
        )
    )
    
    result = response.json()
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    # Extract key information
    if "metadata" in result and "swebench_result" in result["metadata"]:
        # Parse the JSON string
        swebench_result = result["metadata"]["swebench_result"]
        if isinstance(swebench_result, str):
            swebench_result = json.loads(swebench_result)
        metrics = swebench_result.get("swe-bench-metrics", {})
        outputs = swebench_result.get("swe-bench-outputs", {})
        print("\n" + "=" * 40)
        print("EVALUATION RESULTS:")
        print(f"✓ Issue Resolved: {metrics.get('resolved', False)}")
        print(f"✓ Patch Generated: {metrics.get('patch_exists', False)}")
        print(f"✓ Patch Applied: {metrics.get('patch_successfully_applied', False)}")
        print("=" * 40)
        
        # Show patch if generated
        if outputs.get("model_patch"):
            print("\nGENERATED PATCH:")
            print("-" * 40)
            print(outputs["model_patch"])
            print("-" * 40)
        
        # Show trajectory if available
        if "trajectory" in swebench_result:
            print("\nAGENT TRAJECTORY SUMMARY:")
            print("-" * 40)
            trajectory = swebench_result["trajectory"]
            print(trajectory)
    

if __name__ == "__main__":
    print("SWE Agents Client Example")
    print("================================\n")
    print("This example demonstrates solving GitHub issues using AI agents.\n")
    
    asyncio.run(main())
