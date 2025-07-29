# PPO RL Framework with LLM-Sparse Reward
# Step-by-step modules: PolicyNet, EnvWrapper, PPO Loop, LLM Reward Connector

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from GoalBasedEnvironment import GoalBasedEnvironment
import yaml
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
        self.actions = actions  # list of strings
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
            return torch.zeros((1, 1), dtype=torch.long)  # Provide dummy START state
        return torch.tensor(self.current_seq, dtype=torch.long).unsqueeze(0)

    def get_sequence(self):
        return self.current_seq

    def get_action_names(self):
        return [self.actions[a] for a in self.current_seq]

# === 3. LLM Reward Function ===
# === 3. LLM Reward Function ===
def get_llm_reward(sequence_indices, actions):
    import openai
    import json
    OPENAI_API_KEY = ''
    client = OpenAI(api_key = OPENAI_API_KEY)
    action_sequence = [actions[i] for i in sequence_indices]

    # Create the prompt
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

# === 4. PPO Rollout and Update ===
def rollout_episode(policy, env):
    states, actions, log_probs, values = [], [], [], []
    state = env.reset()
    done = False
    while not done:
        logits, value = policy(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        states.append(state.squeeze(0))
        actions.append(action)
        log_probs.append(dist.log_prob(action))
        values.append(value)

        state, done = env.step(action.item())

    sequence = env.get_sequence()
    reward, confidence, explanation, bad_transitions = get_llm_reward(sequence, env.actions)
    return states, actions, log_probs, values, reward, confidence, bad_transitions, explanation

# === 5. Advantage Estimation and PPO Update ===
def ppo_update(policy, optimizer, rollout_data, gamma=0.99, eps_clip=0.2):
    states, actions, log_probs, values, reward, confidence, bad_transitions, _ = rollout_data
    num_transitions = len(values)
    bad_ratio = len(bad_transitions) / max(num_transitions, 1)
    shaped_reward = reward * (1.0 - bad_ratio)

    returns = [shaped_reward * confidence] * num_transitions
    returns = torch.tensor(returns)
    values = torch.stack(values).squeeze()
    log_probs = torch.stack(log_probs)
    actions = torch.stack(actions)

    from torch.nn.utils.rnn import pad_sequence
    states = pad_sequence(states, batch_first=True)

    advantages = returns - values.detach()
    logits, new_values = policy(states)
    new_probs = F.softmax(logits, dim=-1)
    new_dist = torch.distributions.Categorical(new_probs)
    new_log_probs = new_dist.log_prob(actions)

    ratios = torch.exp(new_log_probs - log_probs.detach())
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(new_values, returns)

    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === 6. Training Loop ===
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

    # bag_of_actions = [
    #     "START",
    #     "Add flour to blender",
    #     "Add banana",
    #     "Crack egg",
    #     "Mix well",
    #     "Pour into pan",
    #     "Flip pancake",
    #     "Serve with syrup",
    #     "END"
    # ]

    num_actions = len(bag_of_actions)
    max_seq_len = num_actions


    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    # Instantiate your model and move it to the selected device
    policy = PPOPolicyNet(num_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    env = SequenceEnvironment(bag_of_actions, max_seq_len)

    for episode in range(100):
        rollout_data = rollout_episode(policy, env)
        ppo_update(policy, optimizer, rollout_data)
        if episode % 10 == 0:
            action_names = env.get_action_names()
            print(f"Episode {episode}: Reward = {rollout_data[4]} | Confidence = {rollout_data[5]:.2f}")
            print("Sequence:", " ➜ ".join(action_names))
            print("Bad Transitions:", rollout_data[6])
            print("LLM Explanation:", rollout_data[7])

# Run it
if __name__ == "__main__":
    train()
