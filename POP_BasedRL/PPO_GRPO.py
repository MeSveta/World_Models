# PPO RL Framework with LLM-Sparse Reward
# Step-by-step modules: PolicyNet, EnvWrapper, PPO Loop, LLM Reward Connector

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import openai
import json
import ast
from GoalBasedEnvironment import GoalBasedEnvironment
import yaml
import os
from openai import OpenAI


# === 1. Policy Network ===
class PPOPolicyNet(nn.Module):
    def __init__(self, num_actions, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.action_embeddings = nn.Embedding(num_actions, embedding_dim)
        self.state_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, past_actions):
        embeds = self.action_embeddings(past_actions)              # [B, T, E]
        _, (hidden, _) = self.state_encoder(embeds)                # [1, B, H]
        context = hidden.squeeze(0)                                # [B, H]
        logits = self.policy_head(context)                         # [B, A]
        value = self.value_head(context).squeeze(-1)               # [B]
        return logits, value

# === 2. Environment Wrapper ===
class SequenceEnvironment:
    def __init__(self, actions, max_len):
        self.actions = actions
        self.num_actions = len(actions)
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.current_seq = []
        return self.get_state()

    def step(self, action):
        self.current_seq.append(action)
        done = len(self.current_seq) >= self.max_len
        return self.get_state(), done

    def get_state(self):
        if not self.current_seq:
            return torch.zeros((1, 1), dtype=torch.long)
        return torch.tensor(self.current_seq, dtype=torch.long).unsqueeze(0)

    def get_sequence(self):
        return self.current_seq

    def get_action_names(self):
        return [self.actions[a] for a in self.current_seq]

# === 3. LLM Reward Function ===
def get_llm_reward(sequence_indices, actions):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    action_sequence = [actions[i] for i in sequence_indices]
    prompt = f"""
    You are an expert in verifying procedural action sequences for cooking.
    Given this sequence of actions:
    
    {action_sequence}
    
    Evaluate whether the sequence logically and chronologically leads to completing the task.
    
    Return valid JSON only in this format:
    {{
      "reward": 0 or 1,
      "confidence": float between 0.0 and 1.0,
      "bad_transitions": ["X -> Y", ...],
      "explanation": "..."
    }}
    """
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # Parse the response
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return (
            parsed.get("reward", 0),
            parsed.get("confidence", 0.5),
            parsed.get("explanation", ""),
            parsed.get("bad_transitions", [])
        )
    except Exception as e:
        print("LLM call failed:", e)
        return 0.0, 0.0, "LLM error", []

# === 4. Generate Episodes for Ranking ===
def generate_episodes(policy, env, num_episodes=100):
    episode_data = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode = []
        while not done:
            logits, _ = policy(state)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            episode.append(action)
            state, done = env.step(action)
        episode_data.append(episode)
    return episode_data

# === 5. Rank Episodes ===
def rank_episodes(episodes, actions):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    sequences = [" -> ".join([actions[i] for i in ep]) for ep in episodes]
    prompt = f"""
You are an expert in evaluating cooking procedures.
Here are 100 sequences of actions. Rank them from 0 to 1 based on how likely they are to reach the cooking goal.

Return a list of JSON objects in the following format:
[
  {{"index": 0, "score": 0.75}},
  {{"index": 1, "score": 0.62}},
  ...
]

Sequences:
"""
    for i, seq in enumerate(sequences):
        prompt += f"{i}: {seq}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        ranked = json.loads(content)
        ranked = sorted(ranked, key=lambda x: x["index"])
        return [r["score"] for r in ranked]
    except Exception as e:
        print("LLM batch ranking failed:", e)
        return [0.0] * len(episodes)

# === 6. LSTM Training with LLM Scores ===
class GRPOPolicy(nn.Module):
    def __init__(self, num_actions, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.head(out)

def train_grpo(policy, optimizer, episodes, scores):
    # Pad sequences to equal length for batch training
    max_len = max(len(ep) for ep in episodes) - 1
    padded_inputs = []
    padded_targets = []
    masks = []
    batch_weights = []

    for ep, score in zip(episodes, scores):
        input_seq = ep[:-1]
        target_seq = ep[1:]
        padding = [0] * (max_len - len(input_seq))
        padded_inputs.append(input_seq + padding)
        padded_targets.append(target_seq + padding)
        masks.append([1]*len(input_seq) + [0]*len(padding))
        batch_weights.append(score)

    inputs = torch.tensor(padded_inputs)
    targets = torch.tensor(padded_targets)
    masks = torch.tensor(masks, dtype=torch.float32)
    weights = torch.tensor(batch_weights).view(-1, 1)

    logits = policy(inputs)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # Mask padding tokens
    selected = selected * masks

    # Average loss per sequence, then weight by LLM score
    seq_loss = selected.sum(dim=1) / masks.sum(dim=1)
    weighted_loss = -seq_loss * weights.squeeze()

    loss = weighted_loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === 7. Train Function Overridden ===
def train():
    with open("POP_RL_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    env_config = config['env']
    # #{
    #     "constraints_flag": True,
    #     "reward_type": "standard"  # or "LLM" if using GPTFeedbackConnector
    # }
    json_dir = config['env']['json_path']
    save_dir = config['results']['save_dir']

    if os.path.isdir(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                full_path = os.path.join(json_dir, filename)
            json_path = full_path
    env_1 = GoalBasedEnvironment(env_config, json_path)
    action_dict = env_1.actions

    bag_of_actions = [action_dict[str(i)] for i in range(len(action_dict))]
    num_actions = len(bag_of_actions)
    max_seq_len = num_actions

    env = SequenceEnvironment(bag_of_actions, max_seq_len)
    initial_policy = PPOPolicyNet(num_actions)

    print("Generating episodes...")
    episodes = generate_episodes(initial_policy, env, 10)
    print("Ranking with LLM...")
    scores = rank_episodes(episodes, bag_of_actions)

    print("Training GRPO policy...")
    grpo_policy = GRPOPolicy(num_actions)
    optimizer = torch.optim.Adam(grpo_policy.parameters(), lr=1e-3)
    train_grpo(grpo_policy, optimizer, episodes, scores)

# Run it
if __name__ == "__main__":
    train()
