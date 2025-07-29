# Full DQN setup with LLM reward + discounted Monte Carlo credit assignment

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from transformers import AutoTokenizer, AutoModel
from GPTFeedbackConnector import GPTFeedbackConnector
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import torch.nn.functional as F
import os

EPS_END = 0.05
EPS_DECAY = 1000
EPS_START = 0.9

# --------------------------- Action Set --------------------------- #
actions_start = [
    "START",
    "Add-1/2 tsp baking powder to a blender",
    "Serve the pancakes with chopped strawberries",
    "Melt a small knob of butter in a non-stick frying pan over low-medium heat",
    "Splash maple syrup on plate",
    "Add 1 banana to a blender",
    "Cook for 1 min or until the tops start to bubble",
    "Blitz the blender for 20 seconds",
    "Flip the pancakes with a fork or a fish slice spatula",
    "Add 1 egg to a blender",
    "Cook for 20-30 seconds more",
    "Pour three little puddles straight from the blender into the frying pan",
    "Add 1 heaped tbsp flour to a blender",
    "Chop 1 strawberry",
    "Transfer to a plate",
    "END"
]

steps = {
    "0": "START",
    "1": "Add-1/2 tsp baking powder to a blender",
    "2": "Serve-Serve the pancakes with chopped strawberries",
    "3": "Melt-Melt a small knob of butter in a non-stick frying pan over low-medium heat",
    "4": "splash-splash maple syrup on plate",
    "5": "Add-Add 1 banana to a blender",
    "6": "Cook-Cook for 1 min or until the tops start to bubble",
    "7": "blitz-blitz the blender for 20 seconds",
    "8": "Flip-Flip the pancakes with a fork or a fish slice spatula",
    "9": "Add-1 egg to a blender",
    "10": "cook-cook for 20-30 seconds more",
    "11": "Pour-Pour three little puddles straight from the blender into the frying pan",
    "12": "Add-1 heaped tbsp flour to a blender",
    "13": "Chop-Chop 1 strawberry",
    "14": "Transfer-Transfer to a plate",
    "15": "END"
  }
goal = "preparepancake"
embedding_dim = 768
vocab_size = len(actions_start)

# ------------------ BERT Embedding for Actions ------------------ #
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")
@torch.no_grad()
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert(**inputs)

    return outputs.last_hidden_state[:, 0, :]

action_text_embeddings = torch.cat([get_bert_embedding(t) for t in actions_start], dim=0)
action_text_embeddings = F.normalize(action_text_embeddings, p=2, dim=1)

# ---------------------- Dummy Environment ---------------------- #
class RecipeEnv:
    def __init__(self, bag_of_actions):
        self.max_steps = len(bag_of_actions)
        self.reset()


    def reset(self):
        self.state = [0]  # Always start with "START"
        self.done = False
        self.taken_actions = []
        return self.state.copy()

    def step(self, action):
        reason = []
        if self.done:
            raise Exception("Step called on finished episode.")

        #self.done = len(self.state) >= self.max_steps or action == 15
        if (len(self.state) < self.max_steps and action == 15) or (len(self.state) > self.max_steps and action != 15):
            reward = -7.0/10.0
            self.done = True
        elif action in self.state:
            reward = -5.0/10.0
            reason = ('repetition')
        else:
            self.taken_actions.append(action)
            reward = 0.0
            if len(self.taken_actions)==self.max_steps:
                self.done = True
        # LLM will give reward later
        self.state.append(action)
        return self.state.copy(), reward, self.done, reason



# ------------------- Dummy LLM Reward Function ------------------ #
def get_llm_feedback(trajectory,goal, steps):
    connector = GPTFeedbackConnector()
    reward = connector.evaluate_batch(action_sequence=trajectory, actions=steps,
                                              goal=goal)
    if reward==0:
        reward_llm = -1.0/10.0
    if reward == 1:
        reward_llm = 5.0/10.0
    #texts = [actions[i] for i in trajectory]
    # if any("Serve" in t for t in texts) and any("Transfer" in t for t in texts):
    #     return 1.0
    return reward_llm


# -------------------------- Networks ---------------------------- #
#class StateEncoder(nn.Module):
#     def __init__(self, emb_dim, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
#     def forward(self, x):
#         _, (h, _) = self.lstm(x)
#         return h.squeeze(0)

class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #last_step = x[:, -1, :]# x shape: [1, seq_len, emb_dim]
        return x.mean(dim=1)
        #return last_step

# class DQN(nn.Module):
#     def __init__(self, state_dim, act_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim + act_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.fc:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.1)
#
#     def forward(self, state_emb, act_emb):
#         x = torch.cat((state_emb, act_emb), dim=1)
#         return self.fc(x)

class DQN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, state_emb, act_emb):
        x = torch.cat((state_emb, act_emb), dim=1)
        return self.fc(x)

# -------------------- Replay Buffer (Flat) ---------------------- #
class ReplayBufferFlat:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state_seq, action, reward, next_state_seq, done):
        self.buffer.append((state_seq, action, reward, next_state_seq, done))
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return zip(*samples)
    def __len__(self):
        return len(self.buffer)

def max_argmax(input):
    best_action = np.random.choice([i for i, value in enumerate(input) if value == max(input)])
    return best_action

def normalize(tensor):
    return F.normalize(tensor, p=2, dim=-1)

# -------------------- Training data ---------------------- #
class TrainingMonitor:
    def __init__(self, smoothing_window=20):
        self.episode_rewards = []
        self.loss_log = []
        self.q_log = []
        self.target_log = []
        self.llm_rewards = []
        self.llm_triggered = 0
        self.llm_successes = 0
        self.smoothing_window = smoothing_window

    def log_episode(self, episode_reward, llm_reward=None):
        self.episode_rewards.append(episode_reward)
        if llm_reward is not None:
            self.llm_rewards.append(llm_reward)
            self.llm_triggered += 1
            if llm_reward > 0:
                self.llm_successes += 1

    def log_batch(self, loss, avg_q, avg_target):
        self.loss_log.append(loss)
        self.q_log.append(avg_q)
        self.target_log.append(avg_target)

    def moving_avg(self, data):
        w = self.smoothing_window
        return np.convolve(data, np.ones(w)/w, mode='valid') if len(data) >= w else data

    def plot_all(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Reward per episode
        axs[0, 0].plot(self.moving_avg(self.episode_rewards),'-o')
        axs[0, 0].set_title("Smoothed Episode Reward")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].grid(True)

        # TD Loss
        axs[0, 1].plot(self.moving_avg(self.loss_log),'-o')
        axs[0, 1].set_title("TD Loss Over Time")
        axs[0, 1].set_xlabel("Training Step")
        axs[0, 1].set_ylabel("MSE Loss")
        axs[0, 1].grid(True)

        # Q-values vs Targets
        axs[1, 0].plot(self.moving_avg(self.q_log),'-o', label="Q-values")
        axs[1, 0].plot(self.moving_avg(self.target_log),'-o', label="Targets")
        axs[1, 0].set_title("Q-Value vs Target Value")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # LLM success rate
        if self.llm_triggered > 0:
            success_rate = self.llm_successes / self.llm_triggered
            axs[1, 1].bar(["Success Rate"], [success_rate], color='green' if success_rate > 0 else 'red')
            axs[1, 1].set_ylim(0, 1)
            axs[1, 1].set_title("LLM Success Rate")
            axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

def save_checkpoint(q_network, encoder, optimizer, episode, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"checkpoint_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'q_network': q_network.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)
    print(f"✅ Checkpoint saved at: {filename}")
# ----------------------- Training Loop -------------------------- #
def dqn_training_loop_discounted(env, state_encoder, q_network, optimizer, action_text_embeddings, goal, bag_of_actions,  num_episodes=100, gamma=0.99, batch_size=64):
    monitor = TrainingMonitor()
    replay_buffer_good = ReplayBufferFlat()
    replay_buffer = ReplayBufferFlat(capacity=200)
    replay_buffer_good_llm = ReplayBufferFlat()

    episode_rewards = []
    llm_triggered = 0
    max_buffer_size = 2000


    # fill the buffer with good examples
    permutations_path = 'C:/Users/spaste01/Documents/Research/data/backlog_data/blenderbananapancakes.json'
    with open(permutations_path, 'r') as file:
        data = json.load(file)
    permutations_seq = data['permutations_seq']
    for trajectory in permutations_seq:
        for i in range(len(trajectory) - 1):
            s_seq = trajectory[:i + 1]
            a = trajectory[i + 1]
            ns_seq = trajectory[:i + 2]
            if i == len(trajectory) - 2:
                reward_push = 3.0/10.0
            else:
                reward_push = 1.0/10.0
            replay_buffer_good.push(s_seq, a, reward_push, ns_seq, (i == len(trajectory) - 2))

    for episode in range(num_episodes):
        # if len(replay_buffer) > max_buffer_size:
        #     replay_buffer = copy.deepcopy(replay_buffer_good)
        repetition_found = False
        llm_flag = False
        state = env.reset()
        trajectory = []
        seen_transitions = set()
        done = False

        while not done:
            # # 1. Get list of available (not-yet-taken) actions
            # available_actions = [i for i in range(len(actions_start)) if i not in state]
            # if available_actions==[]:
            #     done= True
            #     continue

            raw_state_emb = state_encoder(action_text_embeddings[torch.tensor(state)].unsqueeze(0))
            state_emb = normalize(raw_state_emb)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * episode / EPS_DECAY)
            if random.random() < eps_threshold:
                #action = random.choice(available_actions)
                action = random.randint(0, len(action_text_embeddings) - 1)
            else:
                #q_vals = [q_network(state_emb, action_text_embeddings[i].unsqueeze(0)) for i in available_actions]
                q_vals = [q_network(state_emb, action_text_embeddings[i].unsqueeze(0)) for i in range(len(action_text_embeddings))]
                # if len(available_actions)==1:
                #     action = available_actions[0]
                # else:
                action = int(max_argmax(torch.stack(q_vals).squeeze()))

            next_state, reward, done, reason = env.step(action)
            if reason=='repetition':
                repetition_found = True
            trajectory.append((state.copy(), action, next_state.copy(),reward, reason))
            state = next_state

        pre_llm_reward = sum([r for (_, _, _, r, _) in trajectory])
        if reward == 0 and  not repetition_found:
            final_reward = get_llm_feedback([a for _, a, _ in trajectory],goal, bag_of_actions)
            llm_triggered += 1
            llm_flag = True
            R = final_reward
        else:
            R = -2.0/10.0

        monitor.log_episode(pre_llm_reward, llm_reward=R if llm_flag else None)

        # episode_rewards.append(R)
        for i in reversed(range(len(trajectory))):

            R = R * gamma
            s_seq, a, ns_seq,r,reason = trajectory[i]
            key = (tuple(s_seq), a, tuple(ns_seq))  # use tuple to make it hashable

            if key in seen_transitions:
                continue
            else:
                seen_transitions.add(key)

            is_terminal = (i == len(trajectory) - 1)
            if llm_flag:
                if R>0:
                    replay_buffer.push(s_seq, a, R, ns_seq, is_terminal)
                    replay_buffer_good.push(s_seq, a, R, ns_seq, is_terminal)
                else:
                    replay_buffer.push(s_seq, a, R, ns_seq, is_terminal)
                    R *= gamma
            else:
                replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            # if reason == 'repetition' :
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            # elif reason ==[] and is_terminal: # Early ending
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            # else:
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            #

        if len(replay_buffer) >= 200:
            # Get samples from both buffers
            states, actions, rewards, next_states, dones = replay_buffer.sample(32)
            states_g, actions_g, rewards_g, next_states_g, dones_g = replay_buffer_good.sample(32)

            states = states+states_g
            actions = actions+actions_g
            rewards = rewards+rewards_g
            next_states = next_states+next_states_g
            dones = dones+dones_g



            # batch1 = list(zip(*replay_buffer.sample(32)))
            # batch2 = list(zip(*replay_buffer_good.sample(32)))
            #
            # # Concatenate each component (e.g., states, actions...)
            # combined = [b1 + b2 for b1, b2 in zip(batch1, batch2)]
            #
            # # Unpack the combined tuples
            # states, actions, rewards, next_states, dones = combined

            state_embeds = torch.cat([
                state_encoder(action_text_embeddings[torch.tensor(s)].unsqueeze(0))
                for s in states
            ], dim=0)
            state_embeds = F.normalize(state_embeds, p=2, dim=1)

            # # LSTM
            # next_state_embeds = torch.cat([
            #     state_encoder(action_text_embeddings[torch.tensor(ns)].unsqueeze(0))
            #     for ns in next_states
            # ], dim=0)

            #Mean pooling
            next_state_embeds = torch.cat([
                state_encoder(action_text_embeddings[torch.tensor(ns)].unsqueeze(0))
                for ns in next_states
            ], dim=0)
            next_state_embeds = F.normalize(next_state_embeds, p=2, dim=1)


            action_embeds = action_text_embeddings[torch.tensor(actions)]
            q_values = q_network(state_embeds, action_embeds).squeeze()

            next_q_values = []
            for ns in next_state_embeds:
                q_vals = [q_network(ns.unsqueeze(0), action_text_embeddings[i].unsqueeze(0)) for i in range(len(action_text_embeddings))]
                next_q_values.append(torch.stack(q_vals).squeeze().max())
            next_q_values = torch.stack(next_q_values)
            #next_q_values = torch.clamp(next_q_values, -1, 1)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            targets = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)



            loss = torch.nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
            params = list(q_network.parameters()) + list(state_encoder.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            # per batch monitor
            monitor.log_batch(loss.item(), q_values.mean().item(), targets.mean().item())

            if episode % 100 == 0:
                state = [0]  # starting state
                state_embedding = encoder(action_text_embeddings[torch.tensor(state)].unsqueeze(0))
                state_embedding = F.normalize(state_embedding, p=2, dim=1)

                q_values = []
                for action_idx in range(len(actions_start)):
                    action_embedding = action_text_embeddings[action_idx].unsqueeze(0)
                    q_val = q_network(state_embedding, action_embedding).squeeze().item()
                    q_values.append(q_val)

                # Plot
                plt.figure(figsize=(10, 5))
                plt.bar(actions_start, q_values, color='skyblue')
                plt.xlabel("Action")
                plt.ylabel("Q-value")
                plt.title("Q-values for Each Action from State [0]")
                plt.xticks(rotation=90)
                plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
            if episode % 10 == 0:
                save_checkpoint(q_network, encoder, optimizer, episode, path="C:/Users/spaste01/PycharmProjects/Results/PPO_RL/checkpoints")
                print("Targets:", targets[:5])
                print("Q-values:", q_values[:5])
                print(f"Episode {episode}: Loss: {loss.item():.4f}, Buffer size: {len(replay_buffer)}, epsilon: {eps_threshold}")

    monitor.plot_all()
    return q_network

# ----------------------- Run the setup -------------------------- #
env = RecipeEnv(bag_of_actions = steps)
# encoder = StateEncoder(embedding_dim, 128)
encoder = StateEncoder()
q_network = DQN(embedding_dim, embedding_dim)
optimizer = optim.Adam(list(q_network.parameters()), lr=1e-3)
q_network = dqn_training_loop_discounted(env, encoder, q_network, optimizer, action_text_embeddings, goal, steps, num_episodes=3000)

#Evaluate Q-values for all actions,
q_values = []
state = [0]
hidden_dim = 128
encoder = StateEncoder()
state_embedding = encoder(action_text_embeddings[torch.tensor(state)].unsqueeze(0))

state = [0,1,5]  # starting state
state_embedding = encoder(action_text_embeddings[torch.tensor(state)].unsqueeze(0))
state_embedding = F.normalize(state_embedding, p=2, dim=1)

q_values = []
for action_idx in range(len(actions_start)):
    action_embedding = action_text_embeddings[action_idx].unsqueeze(0)
    q_val = q_network(state_embedding, action_embedding).squeeze().item()
    q_values.append(q_val)

# Plot
plt.figure(figsize=(10, 5))
plt.bar(actions_start, q_values, color='skyblue')
plt.xlabel("Action")
plt.ylabel("Q-value")
plt.title("Q-values for Each Action from State [0,1]")
plt.xticks(rotation=90)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


def visualize_state_embeddings_tsne(states, state_encoder, action_text_embeddings, title="t-SNE of State Embeddings"):
    """
    states: List of state sequences (each state is a list of token indices)
    state_encoder: your trained StateEncoder model
    action_text_embeddings: precomputed BERT-based embeddings for each action (shape: [vocab_size, emb_dim])
    """
    # Get embeddings
    with torch.no_grad():
        encoded_states = []
        labels = []
        for i, state_seq in enumerate(states):
            emb_seq = action_text_embeddings[torch.tensor(state_seq)].unsqueeze(0)  # shape: [1, seq_len, emb_dim]
            state_emb = state_encoder(emb_seq)  # shape: [1, emb_dim]
            encoded_states.append(state_emb.squeeze().cpu().numpy())
            labels.append(f"{i}: " + " → ".join(str(x) for x in state_seq))  # last 3 actions for label
    # Convert to 2D with t-SNE
    X = torch.tensor(encoded_states)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    X_2d = tsne.fit_transform(X)
    # Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='skyblue', edgecolors='k', s=80)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

