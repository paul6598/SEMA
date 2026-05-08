import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from transformers import pipeline

def main():
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-4b-it",
        device="cuda",
        dtype=torch.bfloat16
    )

    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are the commander of team. You are responsible for giving orders to your agents to achieve the goals. You have access to the observations of the agents and can use them to make informed decisions. You need to coordinate the actions of the agents to achieve the goals in the most efficient way possible. You should consider the constraints of the environment and the capabilities of the agents when making your decisions."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe the following scenario: In this scenario, two agents and two goals are spawned in a narrow corridor. The agents need to reach the goal with their color. The agents are standing in front of each other's goal and thus need to swap places. In the middle of the corridor there is an asymmetric opening which fits one agent only. Therefore the optimal policy is for one agent to give way to the other. This requires heterogeneous behaviour. Each agent observes its position, velocity and the relative position to its goal. The scenario terminates when both agents reach their goals."},
            {"type": "text", "text": "position of agent 1: (0, 0), velocity of agent 1: (0, 0)"},
            {"type": "text", "text": "position of agent 2: (10, 0), velocity of agent 2: (0, 0)"},
            {"type": "text", "text": "at each 10 timestep, I will give you a full obsevation of the environment. and 0 when it reaches its goal."},
            #{"type": "text", "text": "give a simple description of the optimal policy and the reasoning behind it. and do not give a summary of the description of environment and given observations."}
        
        ]
    }
    ]
    response = pipe(messages, max_new_tokens=512, do_sample=False)[0]["generated_text"][-1]["content"]
    vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
    vector = vectorizer.encode(response)
    print(response)
    
                                                                
if __name__ == "__main__":
    main()