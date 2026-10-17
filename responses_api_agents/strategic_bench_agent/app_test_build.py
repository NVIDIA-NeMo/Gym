import os
import json

def get_agent_prompt(config, agent_name, personality, convo_length_limit=15):
    agent_prompt = (
        "This is a negotiation between " + config['agent1_role'] + " and " + config['agent2_role'] +
        ". Here is some general information about the negotiation:\n" + config['generic'] + "\n"
    )

    agent_prompt += "You are " + config[agent_name + '_role'] + ". This is your confidential information:\n" + config[agent_name + '_specific'] + "\n"
    agent_prompt += f"There are a total of {convo_length_limit*2} messages allowed in this conversation (including both parties). You must eventually come to reach an agreement or not reach an agreement within the conversation, and cannot postpone discussion for later. Not reaching an agreement yields no benefit to either party and is heavily discouraged. \n"
    agent_prompt += "Keep responses concise, typically 1-2 sentences per turn and written as natural, realistic dialogue without lists. \n"
    
    mode = personality
    if mode == 'syncophantic':
        agent_prompt += "You are very eager to please the other party and agree with them as much as possible, even at your own expense. You prioritize harmony over your own goals in this negotiation.\n"
    elif mode == 'competitive':
        agent_prompt += "You are very competitive and aim to maximize your own benefit, even at the expense of the other party. You prioritize winning the negotiation over maintaining harmony.\n"
    elif mode == 'balanced':
        agent_prompt += "You aim to balance your own goals with maintaining a good relationship with the other party. You seek mutually beneficial outcomes while also being considerate of the other party's interests.\n"
    # ... We can add the rest of the personalities here ...
    return agent_prompt
