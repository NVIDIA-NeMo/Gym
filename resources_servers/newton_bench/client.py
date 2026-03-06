# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NewtonBench Client Demo - AI Scientist Workflow

This script demonstrates the full workflow of an AI scientist agent
interacting with the NewtonBench resource server:

1. Initialize a session with a physics module (m0_gravity)
2. Run systematic experiments to gather data
3. Analyze data using execute_python (sandbox)
4. Discover the underlying physical law

Prerequisites:
- NewtonBench server running via ng_run
- NewtonBench repo cloned in project root (for run_experiment to work)

Usage:
    python client.py
"""

import asyncio
from typing import Dict

from nemo_gym.server_utils import ServerClient


async def seed_session(
    client: ServerClient,
    cookies: Dict[str, str],
    module_name="m0_gravity",
    difficulty="easy",
    system="vanilla_equation",
    noise_level=0.0,
    law_version="v0",
):
    """
    Initialize a NewtonBench session for a specific physics module.
    """
    response = await client.post(
        server_name="newton_bench",
        url_path="/seed_session",
        json={
            "module_name": module_name,
            "difficulty": difficulty,
            "system": system,
            "noise_level": noise_level,
            "law_version": law_version,
        },
        cookies=cookies,
    )
    response.raise_for_status()
    if response.cookies:
        cookies.update(response.cookies)
    return await response.json()


async def run_gravity_experiment(client: ServerClient, cookies: Dict[str, str], mass1, mass2, distance):
    """
    Run a gravity experiment.
    """
    response = await client.post(
        server_name="newton_bench",
        url_path="/run_experiment_m0_gravity",
        json={
            "mass1": mass1,
            "mass2": mass2,
            "distance": distance,
        },
        cookies=cookies,
    )

    response.raise_for_status()
    return await response.json()


async def execute_python(client: ServerClient, cookies: Dict[str, str], code):
    """
    Execute Python code in the sandboxed environment.
    """
    response = await client.post(
        server_name="newton_bench", url_path="/execute_python", json={"code": code}, cookies=cookies
    )
    response.raise_for_status()
    return await response.json()


async def end_session(client: ServerClient, cookies: Dict[str, str]):
    """
    End the current session and clean up resources.
    """
    response = await client.post(server_name="newton_bench", url_path="/end_session", json={}, cookies=cookies)
    response.raise_for_status()
    return await response.json()


async def demo_physics_discovery():
    """
    Demonstrate the full AI scientist workflow.
    """
    print("=" * 60)
    print("NewtonBench: AI Scientist Demo")
    print("=" * 60)
    print()

    server_client = ServerClient.load_from_global_config()
    cookies: Dict[str, str] = {}

    try:
        print("Phase 1: Starting experiment session...")
        print("   Module: m0_gravity (Newton's Law of Gravitation)")
        print("   Difficulty: easy")
        await seed_session(server_client, cookies, module_name="m0_gravity", difficulty="easy")
        print("   Session initialized")
        print()

        print("Phase 2: Running experiments...")
        print()

        print("   Testing distance dependence (m1=100, m2=100):")
        distances = [10, 20, 40]
        forces_by_distance = []
        for d in distances:
            result = await run_gravity_experiment(server_client, cookies, mass1=100, mass2=100, distance=d)
            res = result.get("result", {})
            if isinstance(res, dict):
                force = res.get("force", res.get("result", 0))
            else:
                force = res
            forces_by_distance.append(force)
            print(f"      distance={d:3} -> force={force:.6e}")
        print()

        print("   Testing mass1 dependence (m2=100, distance=10):")
        masses1 = [50, 100, 200]
        forces_by_mass1 = []
        for m1 in masses1:
            result = await run_gravity_experiment(server_client, cookies, mass1=m1, mass2=100, distance=10)
            res = result.get("result", {})
            if isinstance(res, dict):
                force = res.get("force", res.get("result", 0))
            else:
                force = res
            forces_by_mass1.append(force)
            print(f"      mass1={m1:3} -> force={force:.6e}")
        print()

        print("Phase 3: Analyzing data with Python...")
        print()

        analysis_code = f"""
import numpy as np

# Data from experiments
distances = np.array({distances})
forces_d = np.array({forces_by_distance})
masses1 = np.array({masses1})
forces_m = np.array({forces_by_mass1})

# Check distance dependence: F * r^1.5 should be constant
f_times_r15 = forces_d * distances**1.5
print("Checking power law (F proportional to 1/r^1.5):")
print(f"   F * r^1.5 = {{f_times_r15}}")
ratio = f_times_r15.max() / f_times_r15.min() if f_times_r15.min() != 0 else float('inf')
print(f"   Ratio max/min = {{ratio:.6f}}")
if abs(ratio - 1) < 0.01:
    print("   Confirmed: F is proportional to 1/r^1.5")
else:
    print("   Failed: Does not match 1/r^1.5 law")
print()

# Check mass1 dependence: F / m1 should be constant
f_over_m1 = forces_m / masses1
print("Checking mass1 dependence:")
print(f"   F / m1 = {{f_over_m1}}")
ratio_m = f_over_m1.max() / f_over_m1.min() if f_over_m1.min() != 0 else float('inf')
if abs(ratio_m - 1) < 0.01:
    print("   Confirmed: F is proportional to m1")
print()

# Calculate hidden constant C
# F = C * m1 * m2 / r^1.5
# C = F * r^1.5 / (m1 * m2)
m1, m2, r, F = 100, 100, 10, forces_d[0]
C = F * r**1.5 / (m1 * m2)
print(f"Calculated hidden constant:")
print(f"   C = F * r^1.5 / (m1 * m2) = {{C:.6e}}")
"""

        exec_result = await execute_python(server_client, cookies, analysis_code)
        print("   Sandbox output:")
        if exec_result.get("stdout"):
            for line in exec_result["stdout"].strip().split("\n"):
                print(f"   {line}")
        if exec_result.get("stderr"):
            print("   Errors:")
            for line in exec_result["stderr"].strip().split("\n"):
                print(f"   {line}")
        print()

        print("Phase 4: Discovered Law")
        print()
        print("   Based on experiments and analysis:")
        print("   +----------------------------------------------------------+")
        print("   |  F = C * m1 * m2 / r^1.5                                 |")
        print("   |  (NewtonBench Modified Newton's Law of Gravitation)      |")
        print("   +----------------------------------------------------------+")
        print()

    except Exception as e:
        print(f"ERROR: An error occurred: {e}")
        return
    finally:
        try:
            print("Phase 5: Ending session...")
            await end_session(server_client, cookies)
            print("   Session ended")
            print()
        except Exception:
            pass

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_physics_discovery())
