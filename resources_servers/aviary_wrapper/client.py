#!/usr/bin/env python3
"""
Simple client to test the Aviary HotPotQA wrapper
"""
import asyncio
import json
from nemo_gym.server_utils import ServerClient


async def test_hotpotqa_wrapper():
    """Test the HotPotQA Aviary wrapper"""
    
    server_client = ServerClient.load_from_global_config()
    
    print("Testing Aviary HotPotQA Wrapper")
    print("=" * 40)
    
    # Test 1: Reset environment with a question
    print("\n1. Resetting environment with a question...")
    reset_response = server_client.post(
        server_name="aviary_wrapper_resources_server",
        url_path="/reset",
        json={
            "question": "What is the birth year of Barack Obama?",
            "ground_truth": "1961"
        }
    )
    
    print(f"Reset Response: {json.dumps(reset_response, indent=2)}")
    session_id = reset_response["session_id"]
    
    # Test 2: Search step
    print(f"\n2. Searching for Barack Obama...")
    step_response = server_client.post(
        server_name="aviary_wrapper_resources_server",
        url_path="/step",
        json={
            "session_id": session_id,
            "action_text": "<search>Barack Obama</search>"
        }
    )
    
    print(f"Search Response: {json.dumps(step_response, indent=2)}")
    
    # Test 3: Lookup step
    if not step_response["done"]:
        print(f"\n3. Looking up birth year...")
        step_response = server_client.post(
            server_name="aviary_wrapper_resources_server",
            url_path="/step",
            json={
                "session_id": session_id,
                "action_text": "<lookup>birth year</lookup>"
            }
        )
        
        print(f"Lookup Response: {json.dumps(step_response, indent=2)}")
    
    # Test 4: Submit answer
    if not step_response["done"]:
        print(f"\n4. Submitting answer...")
        step_response = server_client.post(
            server_name="aviary_wrapper_resources_server",
            url_path="/step",
            json={
                "session_id": session_id,
                "action_text": "<submit_answer>1961</submit_answer>"
            }
        )
        
        print(f"Submit Response: {json.dumps(step_response, indent=2)}")
        print(f"Final Reward: {step_response['reward']}")
        print(f"Episode Done: {step_response['done']}")


if __name__ == "__main__":
    asyncio.run(test_hotpotqa_wrapper())