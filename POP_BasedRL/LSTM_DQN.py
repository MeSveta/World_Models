# Full DQN setup with LLM reward + discounted Monte Carlo credit assignment

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from transformers import AutoTokenizer, AutoModel
from POP_BasedRL.GPTFeedbackConnector import GPTFeedbackConnector
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import torch.nn.functional as F
import os
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from collections import Counter
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from collections import defaultdict
from POP_BasedRL.utils.collect_results import TrajectoryLogger
from POP_BasedRL.utils.Metrics import Metrics

EPS_END = 0.05
EPS_DECAY = 1000
EPS_START = 0.9

MAX_SEQ_LEN = 4

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
steps = {str(i): action for i, action in enumerate(actions_start)}
goal = "preparepancake"
embedding_dim = 768
vocab_size = len(actions_start)
physics_data = {}
edges_llm = []


input_path = r"C:\Users\spaste01\Documents\Research\data\train_data_llm_proxy\blenderbananapancakes.json"
with open(input_path, 'r') as file:
    data = json.load(file)
#data_physics = data['physics']# Merge both dictionaries
world_model = data.get("LLM_costarins_with_physics_llm_proxy_4o_mini", {})
edges = data.get("edges")
for target_str, data_w in world_model.items():
    target = int(target_str)
    for source_str in data_w["preconditions"]:
        source = int(source_str)
        edges_llm.append((source, target))

# 1. Convert keys to integers and gather all edges from preconditions
preconditions = defaultdict(list)
for target_str, data in world_model.items():
    target = int(target_str)
    for source_str in data["preconditions"]:
        source = int(source_str)
        preconditions[target].append(source)

#convert it to a regular dict (not defaultdict):
preconditions = dict(preconditions)

metric = Metrics(true_constrains=edges)
metric_llm = Metrics(true_constrains=edges_llm)

merged = {}
# for key in steps:
#     physics_data[key] = {
#         "Effect": data_physics[key]["Effect"],
#         "Irreversible": data_physics[key]["Irreversible"]
#     }

# ------------------ BERT Embedding for Actions ------------------ #
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")
bert.eval()
@torch.no_grad()
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :]

action_text_embeddings = torch.cat([get_bert_embedding(t) for t in actions_start], dim=0)
action_text_embeddings = F.normalize(action_text_embeddings, p=2, dim=1)
goal_text_embedding = torch.tensor(get_bert_embedding(goal))
goal_text_embedding = F.normalize(goal_text_embedding, p=2, dim=1)


class PlanEmbeddingScorer:
    def __init__(self, action_texts, action_embeddings, goal_embedding=None):
        """
        action_texts: list of str (action descriptions)
        action_embeddings: torch.Tensor of shape (N, D)
        goal_embedding: torch.Tensor of shape (D,) or (1, D)
        """
        self.action_texts = action_texts
        self.embeddings = action_embeddings
        self.goal_embedding = goal_embedding.squeeze() if goal_embedding is not None else None
        self.num_actions = len(action_texts)
        self.score_matrix = None

    def compute_transition_scores(self):
        """
        Computes a matrix of transition scores between all pairs of actions.
        If goal is provided, uses goal alignment. Else uses transition coherence.
        """
        scores = torch.zeros((self.num_actions, self.num_actions))
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                if i == j:
                    scores[i, j] = float('nan')
                    continue
                delta = self.embeddings[j] - self.embeddings[i]
                if self.goal_embedding is not None:
                    goal_vec = self.goal_embedding - self.embeddings[i]
                    score = F.cosine_similarity(delta.unsqueeze(0), goal_vec.unsqueeze(0), dim=1).item()
                else:
                    score = F.cosine_similarity(self.embeddings[i].unsqueeze(0), self.embeddings[j].unsqueeze(0), dim=1).item()
                scores[i, j] = score
        self.score_matrix = scores
        return scores

    def score_trajectory(self, trajectory_indices):
        """
        Scores a trajectory based on cosine coherence between consecutive steps.
        """
        scores = []
        for i in range(len(trajectory_indices) - 1):
            a = self.embeddings[trajectory_indices[i]]
            b = self.embeddings[trajectory_indices[i + 1]]
            score = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def plot_transition_heatmap(self, figsize=(10, 8)):
        if self.score_matrix is None:
            self.compute_transition_scores()
        np_scores = self.score_matrix.numpy()
        plt.figure(figsize=figsize)
        sns.heatmap(
            np_scores,
            xticklabels=self.action_texts,
            yticklabels=self.action_texts,
            annot=False,
            cmap="viridis",
            mask=np.isnan(np_scores)
        )
        plt.title("Transition Plausibility Heatmap")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# ---------------------- Dummy Environment ---------------------- #
class RecipeEnv:
    def __init__(self, bag_of_actions, preconditions):
        self.max_steps = len(bag_of_actions)
        self.reset()
        self.action_mask = []
        self.preconditions = preconditions

    def reset(self):
        self.state = [0]  # Always start with "START"
        self.done = False
        self.taken_actions = []
        self.num_steps = 0
        return self.state.copy()

    def step(self, action):
        reason = []

        if self.done:
            raise Exception("Step called on finished episode.")

        if (self.num_steps < self.max_steps-2 and action == 15) or (self.num_steps == self.max_steps-2 and action != 15):
            reward = -1
            self.done = True
        elif action in self.state:
            reward = -2.0/10.0
            reason = ('repetition')
            self.done = True
        # elif not all(p in self.state for p in self.preconditions[action]):
        #     reward = -2.0 / 10.0
        #     reason = ('precondition')
        #     self.done = True
        else:
            self.taken_actions.append(action)
            reward = 0.0
            if len(self.taken_actions) == self.max_steps-1:
                self.done = True
        self.num_steps += 1
        self.state.append(action)
        return self.state.copy(), reward, self.done, reason

# ------------------- Dummy LLM Reward Function ------------------ #
def get_llm_feedback(trajectory, full_episode, goal, steps, world_model, metric,metric_llm, results_log_LLM, llm_i):
    connector = GPTFeedbackConnector()
    r = []
    r_parts = []
    temp_ex_list = []
    #trajectory = [0, 5, 7, 1, 9, 13, 12, 3, 11, 8, 6, 10, 14, 2, 4, 15]
    #trajectory = [0, 3, 13, 5, 1, 9, 12, 7, 11, 6, 8, 10, 14, 2, 4, 15]
    #trajectory = [0, 5, 1, 3, 9, 12, 7, 13, 11, 4, 8, 10, 6, 2, 14, 15]
    effects = {}

    reward_transition = None
    reward_effects = None
    state_trace = None
    response = None
    reward_contrastive = None
    reward_contrastive_cons = None
    valid_sequence_1 = [0, 13, 5, 1, 3, 9, 12, 7, 11, 6, 8, 10, 14, 2, 4, 15]
    valid_sequence_2 = [0, 5, 1, 3, 9, 12, 7, 13, 11, 6, 8, 10, 14, 2, 4, 15]
    invalid_sequence_1 =  [0, 5, 7, 1, 9, 13, 12, 3, 11, 8, 6, 10, 14, 2, 4, 15]
    invalid_sequence_2 =  [0, 5, 1, 3, 9, 12, 7, 13, 11, 4, 8, 10, 6, 2, 14, 15]

    llm_contrains_eval = metric_llm.check_trajectory_edges(trajectory)

    while reward_transition == None:
        reward_transition, explanation_transition = connector.evaluate_transition(action_sequence=trajectory,
                                                                                  actions=steps, goal=goal)
    # while reward_effects == None:
    #     reward_effects, explanation_effects = connector.evaluate_transition_and_effects(action_sequence=trajectory, actions=steps,
    #                                                                      goal=goal, effects=effects)
    while state_trace==None:
        state_trace = connector.state_evaluation_no_world_model(action_sequence=trajectory, actions=steps, goal=goal)

    while response==None:
        response = connector.eval_llm_based_constrains_and(action_sequence=trajectory, actions=steps, goal=goal,
                                                           violated_subsequences=llm_contrains_eval['violated_in_current_sequence'], state_trace = state_trace)
       #connector.state_evaluation_no_world_model(action_sequence=trajectory, actions=steps, goal=goal)
    while reward_contrastive==None:
        reward_contrastive, explanation_contrastive = connector.evaluate_contrastive(action_sequence=trajectory,
                                                                                  actions=steps, goal=goal,valid_sequence_1=valid_sequence_1, valid_sequence_2=valid_sequence_2,invalid_sequence_1=invalid_sequence_1, invalid_sequence_2=invalid_sequence_2)

    while reward_contrastive_cons==None:
        reward_contrastive_cons, explanation_contrastive_cons = connector.evaluate_contrastive_violation_constrains(action_sequence=trajectory,
                                                                                  actions=steps, goal=goal,violated_subsequences=llm_contrains_eval['violated_in_current_sequence'], valid_sequence_1=valid_sequence_1, valid_sequence_2=valid_sequence_2,invalid_sequence_1=invalid_sequence_1, invalid_sequence_2=invalid_sequence_2)
    result = metric.check_trajectory_edges(trajectory)
    entry,entry_id = results_log_LLM.new_entry(trajectory, gt_label=int(result['all_respected']), reward_transition=reward_transition, explanation_transition = explanation_transition, reward_contrastive = reward_contrastive, explanation_contrastive=explanation_contrastive, reward_contrastive_cons = reward_contrastive_cons, explanation_contrastive_cons=explanation_contrastive_cons,reward_cons_state = response['reward'], explanation_cons_state = response['explanation'] ,state_trace=state_trace)
    # entry = results_log_LLM.update_direct_eval(entry, entry_id, reward=reward_w, explanation=explanation_w)
    # entry = results_log_LLM.update_state_eval(entry, entry_id, state_trace=parsed_states, reward=reward, explanation=explanation)
    results_log_LLM.log_entry(entry_id, entry)
    results_log_LLM.save()


    if reward_contrastive_cons == 1:
        return 7.0/10.0,explanation_contrastive_cons
    else:
        return -1.0,explanation_contrastive_cons

# -------------------------- Networks ---------------------------- #
class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMDQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, lengths):
        # Pack the sequence for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        # Use the last hidden state
        out = h_n[-1]
        q_values = self.fc(out)
        return q_values


class LSTMDQN_Mask(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, output_dim):
        super(LSTMDQN_Mask, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim+num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, action_mask, lengths):
        # Pack the sequence for efficient processing
        # Concatenate state emb, action mask, and action emb

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        # Use the last hidden state
        out = h_n[-1]

        action_mask_tensor = torch.stack(action_mask)
        SCALE = 0.1  # tune this empirically
        scaled_mask = action_mask_tensor.squeeze(1) * SCALE
        combined = torch.cat([out, scaled_mask], dim=1)
        q_values = self.fc(combined)
        return q_values


# class LSTMDQN_Mask(nn.Module):
#     def __init__(self, input_dim, hidden_dim, action_dim, num_actions):
#         super(LSTMDQN_Mask, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.action_proj = nn.Linear(action_dim, hidden_dim)
#         #self.lstm = nn.LSTM(input_dim, 128, batch_first=True)  # was 128
#
#         # Now input is: [LSTM hidden + action mask + action embedding]
#         self.q_head = nn.Sequential(
#             nn.Linear(hidden_dim*2 + num_actions, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#             )
#
#         # self.q_head = nn.Sequential(
#         #     nn.Linear(hidden_dim + action_dim + num_actions, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, 256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, 1),
#         # )
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.q_head:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.1)
#
#     def forward(self, state_seq_emb, action_emb, action_mask):
#         """
#         state_seq_emb: [batch, seq_len, emb_dim]  - last T actions embeddings
#         action_emb:    [batch, emb_dim]            - candidate action embedding
#         action_mask:   [batch, num_actions]        - occurrence mask (0/1 per action)
#         """
#         _, (h_n, _) = self.lstm(state_seq_emb)  # h_n: [1, batch, hidden]
#         state_emb = h_n.squeeze(0)
#
#         action_emb_proj = self.action_proj(action_emb) # [batch, hidden]
#
#         # Concatenate state emb, action mask, and action emb
#         SCALE = 0.1  # tune this empirically
#         scaled_mask = action_mask * SCALE
#         x = torch.cat((state_emb, action_emb_proj, action_mask), dim=1)
#         return self.q_head(x)

    # -------------------- Replay Buffer (Flat) ---------------------- #
class ReplayBufferFlat:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state_seq, action, reward, next_state_seq, action_mask, action_mask_n, lengths_s, lengths_ns, done):
        self.buffer.append((state_seq, action, reward, next_state_seq, action_mask, action_mask_n, lengths_s, lengths_ns, done))
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return zip(*samples)
    def get_last_buffer(self, n=2000):
        new_buffer = ReplayBufferFlat(capacity=n)
        last_n = list(self.buffer)[-n:]
        new_buffer.buffer.extend(last_n)
        return new_buffer
    def contains(self, s_seq, a):
        key_to_check = (tuple(s_seq), a)
        for state_seq, action, reward,n_state,action_mask, action_mask_n, lengths_s, lengths_ns, done in self.buffer:
            if (tuple(state_seq), action) == key_to_check:
                return True,state_seq, action, reward,n_state,action_mask, action_mask_n, lengths_s, lengths_ns, done
        if len(self.buffer) == 0:
            return False, [], [], [], [], [], [],[],[],[]
        else:
            return False,state_seq, action, reward,n_state,action_mask,action_mask_n, lengths_s, lengths_ns, done
    def __len__(self):
        return len(self.buffer)

def max_argmax(input):
    best_action = np.random.choice([i for i, value in enumerate(input) if value == max(input)])
    return best_action

def normalize(tensor):
    return F.normalize(tensor, p=2, dim=-1)

# -------------------- Training data ---------------------- #
def prepare_lstm_batch(sequences, action_text_embeddings):
    """Convert a list of token index sequences into padded tensor batch for LSTM input."""
    tensors = [action_text_embeddings[torch.tensor(seq)] for seq in sequences]
    padded = pad_sequence(tensors, batch_first=True)  # [B, max_len, D]
    lengths = torch.tensor([len(seq) for seq in sequences])
    return padded, lengths

def iterate_batches(*buffers, batch_size,train_size=2000):
    """
    Takes any number of ReplayBufferFlat instances and yields batches from the combined data.
    """
    full = []
    for buf in buffers:
        full.extend(list(buf.buffer))
        l_len= len(full)
    # full_last = full[l_len-train_size:l_len]
    random.shuffle(full)
    for i in range(0, len(full), batch_size):
        yield zip(*full[i:i+batch_size])
class TrainingMonitor:
    def __init__(self, save_path, smoothing_window=100):
        self.episode_rewards = []
        self.loss_log = []
        self.q_log = []
        self.target_log = []
        self.llm_rewards = []
        self.llm_triggered = 0
        self.llm_successes = 0
        self.smoothing_window = smoothing_window
        self.save_path = save_path

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

    def plot_all(self,episode_num):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Reward per episode
        axs[0, 0].plot(self.moving_avg(self.episode_rewards),'-')
        axs[0, 0].set_title("Smoothed Episode Reward")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].grid(True)

        # TD Loss
        axs[0, 1].plot(self.moving_avg(self.loss_log),'-')
        axs[0, 1].set_title("TD Loss Over Time")
        axs[0, 1].set_xlabel("Training Step")
        axs[0, 1].set_ylabel("MSE Loss")
        axs[0, 1].grid(True)

        # Q-values vs Targets
        axs[1, 0].plot(self.moving_avg(self.q_log),'-', label="Q-values")
        axs[1, 0].plot(self.moving_avg(self.target_log),'-', label="Targets")
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
        filename = f"ep{episode_num}.png"
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        plt.show()

def save_checkpoint(q_network, optimizer, episode, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"checkpoint_ep{episode}.pth")
    torch.save({
        'episode': episode,
        'q_network': q_network.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)
    print(f"✅ Checkpoint saved at: {filename}")

def visualize_state_embeddings_tsne(states, q_network, action_mask, action_text_embeddings, episode_num, save_path, title="t-SNE of State Embeddings"):
    """
    states: List of state sequences (each state is a list of token indices)
    q_network: your trained LSTM-based Q-network
    action_text_embeddings: precomputed BERT-based embeddings for each action (shape: [vocab_size, emb_dim])
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    # Get embeddings
    with torch.no_grad():
        encoded_states = []
        labels = []
        for i, state_seq in enumerate(states):
            emb_seq = action_text_embeddings[torch.tensor(state_seq)].unsqueeze(0)  # [1, seq_len, emb_dim]
            emb_seq = F.normalize(emb_seq, p=2, dim=2)
            # SCALE = 0.1  # tune this empirically
            # scaled_mask = action_mask * SCALE
            # x = torch.cat((emb_seq, scaled_mask), dim=1)
            _, (h_n, _) = q_network.lstm(emb_seq)
            state_emb = h_n.squeeze(0)
            state_emb = state_emb.squeeze(0)# [1, hidden_dim] -> [hidden_dim]
            encoded_states.append(state_emb.cpu().numpy())
            labels.append(f"{i}: " + " → ".join(str(x) for x in state_seq))

    # Convert to 2D with t-SNE
    X = torch.tensor(encoded_states)
    tsne = TSNE(n_components=2, perplexity=min(10, len(states) - 1), random_state=42)
    X_2d = tsne.fit_transform(X)

    # Plot
    plt.title(f"Episode {episode_num} - t-SNE of State Embeddings")
    plt.figure(figsize=(12, 7))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='skyblue', edgecolors='k', s=80)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    filename = f"ep{episode_num}_t-SNE of State Embeddings.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def q_value_entropy(q_values, temperature=1.0):
    # Convert Q-values to a probability distribution
    probs = F.softmax(q_values / temperature, dim=0)  # shape: [num_actions]

    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-8))

    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs).item()
    return entropy

def compute_transition_alignment(action_embeddings, goal_vector, actions_dict, visualize=True):
    """
    Computes a transition plausibility matrix by comparing (B - A) and (Goal - A) vectors.

    Args:
        action_embeddings (dict): {action_id: torch.tensor} with shape (768,)
        goal_vector (torch.tensor): goal embedding, shape (768,)
        actions_dict (dict): {action_id: action_text} for labels
        visualize (bool): whether to display a heatmap

    Returns:
        score_matrix (np.ndarray): [N x N] matrix of cosine similarity scores
    """
    n = len(action_embeddings)
    score_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                a = action_embeddings[i]
                b = action_embeddings[j]
                transition_vec = b - a
                goal_vec = goal_vector - a
                score = F.cosine_similarity(
                    transition_vec.unsqueeze(0),
                    goal_vec
                ).item()
                score_matrix[i, j] = score
            else:
                score_matrix[i, j] = np.nan  # mask diagonal

    if visualize:
        plt.figure(figsize=(12, 10))
        sns.heatmap(score_matrix, xticklabels=[actions_dict[i].split()[0] for i in range(n)],
                    yticklabels=[actions_dict[i].split()[0] for i in range(n)],
                    cmap="coolwarm", annot=False, mask=np.isnan(score_matrix))
        plt.title("Transition Plausibility: cos((B−A), Goal−A)")
        plt.xlabel("To Action (B)")
        plt.ylabel("From Action (A)")
        plt.tight_layout()
        plt.show()

    return score_matrix

def visualize_policy_and_q_values(env, q_network, episode_num, save_path="q_value_plots"):
    # state = env.reset()  # Start from initial state
    # done = False
    # step = 0
    # policy = [0]
    # num_actions = env.max_steps
    # action_mask = torch.zeros(1, len(action_text_embeddings))
    # action_mask[0, 0] = 1  # allways starts with START state

    for i in range(2):
        state = env.reset()  # Start from initial state
        done = False
        step = 0
        policy = [0]
        num_actions = env.max_steps
        action_mask = torch.zeros(1, len(action_text_embeddings))
        action_mask[0, 0] = 1  # allways starts with START state
        while not done and step < num_actions:
            # Get Q-values from the network
            state_seq_emb = action_text_embeddings[torch.tensor(state)].unsqueeze(0)
            action_mask[0, state] = 1.0
            lengths_s = torch.tensor([len(state)])
            #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
            q_values = q_network(state_seq_emb, (action_mask,), lengths_s).detach().squeeze()
            # [num_actions]
            q_entropy = q_value_entropy(q_values, temperature=1.0)
            # Plot Q-values
            plt.figure()
            plt.bar(range(num_actions), q_values.numpy())
            plt.title(f"i{i} Episode {episode_num} - Step {step} - State {state} -  Entropy:{q_entropy}")
            plt.xlabel("Action")
            plt.ylabel("Q-value")
            #plt.show()
            # Save figure
            filename = f"i{i}_ep{episode_num}_step{step}_statelen{len(state)}.png"
            plt.savefig(os.path.join(save_path, filename))
            plt.close()

            if i==1 and step==0:
                # Get top 2 values and their indices
                top2_vals, top2_indices = torch.topk(q_values, 2)

                best_action = top2_indices[1].item()
            else:
                # Choose the best action
                best_action = torch.argmax(q_values).item()
                policy.append(best_action)
            print(f"Step {step}: State {state} -> Action {best_action}")

            # Apply action
            next_state, reward, done, _ = env.step(best_action)
            if len(next_state) <= MAX_SEQ_LEN:
                state = next_state
            else:
                state = next_state[-MAX_SEQ_LEN:]
            #state = next_state
            step += 1

        print("\nFinal Policy: ", policy)

# ----------------------- Training Loop -------------------------- #
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def dqn_training_loop_discounted(env, q_network, target_network, optimizer, action_text_embeddings, goal, goal_text_embedding, bag_of_actions,  metric, num_episodes=100, gamma=0.99, batch_size=32):
    max_buffer_size = 2000
    max_num_episodes_first_part = 2000
    monitor = TrainingMonitor(save_path = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/PLOTS/plots_mask_retrain_physics/llmproxy_world_models_gpt_4o_mini/")
    results_log_LLM = TrajectoryLogger(json_path="C:/Users/spaste01/PycharmProjects/Results/PPO_RL/LLM_evaluation_trajectories/blenderbananapancakes.json")

    replay_buffer_good = ReplayBufferFlat()
    replay_buffer = ReplayBufferFlat(capacity=max_buffer_size)
    replay_buffer_good_llm = ReplayBufferFlat()
    action_mask = torch.zeros(1, len(action_text_embeddings))
    llm_good_episodes = {}
    llm_good_episodes_list = []

    episode_rewards = []
    llm_triggered = 0
    llm_i = 0
    target_update_freq = 10

    # buffer path
    save_path_llm = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/BUFFERS_total/buffers_llmproxy_gpt_4o_mini_world_models/replay_buffer_good_llm_cur.pkl"
    save_path_llm_full_episodes = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/BUFFERS_total/buffers_llmproxy_gpt_4o_mini_world_models/good_llm_full_episodes.pkl"
    save_path_buffer = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/BUFFERS_total/buffers_llmproxy_gpt_4o_mini_world_models/replay_buffer_cur_part2.pkl"
    save_path_buffer_2 = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/BUFFERS_total/buffers_llmproxy_gpt_4o_mini_world_models/replay_buffer_cur_part_next.pkl"
    save_path_buffer_load = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/BUFFERS_total/buffers_llmproxy_gpt_4o_mini_world_models/replay_buffer_cur.pkl"


    # with open(save_path_buffer_load, "rb") as f:
    #     replay_buffer_part_1 = pickle.load(f)
    with open(save_path_buffer, "rb") as f:
         replay_buffer = pickle.load(f)


    # with open(save_path_llm_full_episodes, "rb") as f:
    #      replay_buffer_llm= json.load(f)


    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_llm), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_buffer), exist_ok=True)


    # fill the buffer with good examples
    permutations_path = 'C:/Users/spaste01/Documents/Research/data/backlog_data/blenderbananapancakes.json'
    with open(permutations_path, 'r') as file:
        data = json.load(file)
    permutations_seq = data['permutations_seq']
    for trajectory in permutations_seq:
        action_mask = torch.zeros(1, len(action_text_embeddings))
        action_mask_n = torch.zeros(1, len(action_text_embeddings))
        for i in range(len(trajectory) - 1):
            # Truncate state and next state to the last MAX_SEQ_LEN actions
            s_seq = trajectory[max(0, i + 1 - MAX_SEQ_LEN):i + 1]  # up to i (inclusive)
            ns_seq = trajectory[max(0, i + 2 - MAX_SEQ_LEN):i + 2]  # up to i+1 (inclusive)
            lengths_s = torch.tensor([len(s_seq)])
            lengths_ns = torch.tensor([len(ns_seq)])

            a = trajectory[i + 1]
            action_mask[0, s_seq] = 1.0
            action_mask_n[0,ns_seq] = 1.0

            # Assign higher reward if this is the final step
            reward_push = 0.7 if i == len(trajectory) - 2 else 0.5
            done = (i == len(trajectory) - 2)
            contains_flag = replay_buffer_good.contains(s_seq, a)

            if not contains_flag[0]:
                replay_buffer_good.push(s_seq, a, reward_push, ns_seq, action_mask.clone(),action_mask_n.clone(), lengths_s, lengths_ns, done)


    for episode in range(num_episodes):
        episode = episode+1800
        # if len(replay_buffer) > max_buffer_size:
        #     replay_buffer = copy.deepcopy(replay_buffer_good)
        repetition_found = False
        llm_flag = False
        state = env.reset()
        trajectory = []
        seen_transitions = set()
        done = False
        action_mask = torch.zeros(1, len(action_text_embeddings))
        action_mask[0,0] = 1 #allways starts with START state
        action_mask_n = torch.zeros(1, len(action_text_embeddings))
        action_embeddings_dict = {
            idx: embedding for idx, embedding in enumerate(action_text_embeddings)
        }

        if episode<=max_num_episodes_first_part:
            value = 1
            # Save the buffer
            # with open(save_path_buffer, "wb") as f:
            #     pickle.dump(replay_buffer, f)
        else:
            value = random.randint(0, 1)
            save_path_buffer = save_path_buffer_2
        while not done:
            state_seq_emb = action_text_embeddings[torch.tensor(state)].unsqueeze(0)
            lengths_s = torch.tensor([len(state)])

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * episode / EPS_DECAY)
            if episode>=max_num_episodes_first_part:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * (episode-max_num_episodes_first_part) / EPS_DECAY)
            #eps_threshold = 1
            action_mask[0, state] = 1.0
            q_vals = q_network(state_seq_emb, (action_mask,), lengths_s)
            # 1. Get indices of available actions (mask == 0)
            action_mask_temp = action_mask.squeeze(0).long()
            available_indices = torch.nonzero(action_mask_temp == 0, as_tuple=True)[0]  # shape: [num_available]

            # 2. Get indices of positive Q-values
            positive_mask = q_vals.squeeze() > 0
            positive_indices = torch.nonzero(positive_mask, as_tuple=True)[0]  # shape: [num_positive]

            # 3. Intersection: only actions that are both available and have positive Q
            intersection = torch.tensor(list(set(available_indices.tolist()) & set(positive_indices.tolist())))

            # choose randomly if to run with permutation constrains value=0 or to panalize value=1 (it should be both ), since the agent should remebber not to repiet

            if random.random() < eps_threshold:
                if value==1:
                    action = random.randint(0, len(action_text_embeddings) - 1)
                    if episode % 100 == 0:
                        # Save the buffer
                        with open(save_path_buffer, "wb") as f:
                            pickle.dump(replay_buffer, f)
                else:

                    if intersection.numel() > 0:
                        action = intersection[torch.randint(len(intersection), (1,))].item()
                    elif available_indices.numel()==0:
                        if positive_indices.numel() > 0:
                            action = positive_indices[torch.randint(len(positive_indices), (1,))].item()
                        else:
                            action = random.randint(0, len(action_text_embeddings) - 1)
                    else:
                        # fallback (e.g., choose best available)
                        fallback_action = available_indices[torch.randint(len(available_indices), (1,))].item()
                        action = fallback_action

            else:
                #q_vals = q_network(state_seq_emb,lengths_s)
                if value==1:
                    action = torch.argmax(q_vals, dim=1).item()
                else:
                    if available_indices.numel()>0:
                        fallback_action = available_indices[torch.argmax(q_vals.squeeze()[available_indices])].item()
                        action = fallback_action
                    else:
                        action = torch.argmax(q_vals, dim=1).item()


                # q_vals = q_network(state_seq_emb, (action_mask,),lengths_s)
                # action = torch.argmax(q_vals, dim=1).item()


            env.action_mask = action_mask
            next_state, reward, done, reason = env.step(action)

            action_mask_n[0, next_state] = 1.0
            if len(next_state)<=MAX_SEQ_LEN:
                lengths_ns = torch.tensor([len(next_state)])
            else:
                lengths_ns = torch.tensor([len(next_state[-MAX_SEQ_LEN:])])

            if reason=='repetition':
                repetition_found = True
            trajectory.append((state.copy(), action, next_state[-MAX_SEQ_LEN:].copy(),reward, action_mask.clone(), action_mask_n.clone(), lengths_s, lengths_ns, reason))
            if len(next_state)<=MAX_SEQ_LEN:
                state = next_state
            else:
                state = next_state[-MAX_SEQ_LEN:]


        pre_llm_reward = sum([r for (_, _, _, r, _, _, _,_,_) in trajectory])


        if reward == 0 and not repetition_found:
            sequence_llm = [s[-1] for s, _, _,_,_,_,_,_,_ in trajectory]
            _, a, _, _, _, _, _, _, _ = trajectory[-1]
            sequence_llm.append(a)
            #compute_transition_alignment(action_embeddings_dict, goal_text_embedding, [], visualize=False)
            scorer = PlanEmbeddingScorer(actions_start, action_embeddings_dict, goal_text_embedding)
            score_matrix = scorer.compute_transition_scores()
            trajectory_score = scorer.score_trajectory(sequence_llm)
            if not sequence_llm in permutations_seq and not sequence_llm in llm_good_episodes_list:
                final_reward,explanation = get_llm_feedback(sequence_llm, trajectory, goal, bag_of_actions, world_model, metric, metric_llm, results_log_LLM, llm_i)
                llm_triggered += 1
                llm_flag = True
                R = final_reward
                if R>0:
                    if sequence_llm not in llm_good_episodes_list:
                        llm_good_episodes[llm_i] = {"episode": sequence_llm, "explanation":explanation}
                        llm_i += 1
                        llm_good_episodes_list.append(sequence_llm)
                    # Save to file
                    with open(save_path_llm_full_episodes, "w") as f:
                        json.dump(llm_good_episodes, f, indent=2)
        else:
            R = -2.0/10.0

        monitor.log_episode(pre_llm_reward, llm_reward=R if llm_flag else None)

        # episode_rewards.append(R)
        for i in reversed(range(len(trajectory))):

            s_seq, a, ns_seq,r, action_mask, action_mask_n, lengths_s, lengths_ns, reason = trajectory[i]
            key = (tuple(s_seq), a, tuple(ns_seq))  # use tuple to make it hashable

            if key in seen_transitions:
                continue
            else:
                seen_transitions.add(key)

            is_terminal = (i == len(trajectory) - 1)
            contains_flag = replay_buffer_good.contains(s_seq, a)
            contains_flag_llm = replay_buffer_good_llm.contains(s_seq, a)
            if llm_flag:
                if R>0:
                    if not contains_flag[0] or not contains_flag_llm[0]:
                        if i == len(trajectory) - 1:
                    #replay_buffer.push(s_seq, a, R, ns_seq, action_mask.clone(), is_terminal)
                            replay_buffer_good_llm.push(s_seq, a, R, ns_seq, action_mask.clone(), action_mask_n.clone(),lengths_s, lengths_ns, is_terminal)
                        else:
                            replay_buffer_good_llm.push(s_seq, a, 0.5, ns_seq, action_mask.clone(), action_mask_n.clone(),lengths_s, lengths_ns, is_terminal)

                        # Save the buffer
                        with open(save_path_llm, "wb") as f:
                            pickle.dump(replay_buffer_good_llm, f)
                        # with open(save_path_llm_full_episodes, "wb") as f:
                        #     pickle.dump(sequence_llm, f)

                else:
                    if i == len(trajectory) - 1:
                        r = R
                    else:
                        r += R * gamma ** (len(trajectory) - 1 - i)
            else:
                if i == len(trajectory) - 1:
                    R = r
                else:
                    r += R*gamma**(len(trajectory)-1-i)
                if not contains_flag[0]:
                    replay_buffer.push(s_seq, a, r, ns_seq, action_mask.clone(), action_mask_n.clone(),lengths_s, lengths_ns, is_terminal)
            # if reason == 'repetition' :
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            # elif reason ==[] and is_terminal: # Early ending
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            # else:
            #     replay_buffer.push(s_seq, a, r, ns_seq, is_terminal)
            #
        #replay_buffer_good_temp = deque(list(replay_buffer_good.buffer)[:2])
        l_buffer = len(replay_buffer)
        if l_buffer >= max_buffer_size:
            # Get samples from both buffers
            #for states, actions, rewards, next_states, action_mask, action_mask_n, lengths_s, lengths_ns, dones in replay_buffer_good_temp:

            #buffer_part = replay_buffer.get_last_buffer()
            for states, actions, rewards, next_states, action_mask, action_mask_n, lengths_s, lengths_ns, dones in iterate_batches(
                    replay_buffer,replay_buffer_good,replay_buffer_good_llm, batch_size=batch_size):
                # Now use these batch variables for your training step


                # Encode all state sequences
                state_embeds, lengths = prepare_lstm_batch(states, action_text_embeddings)


                # # LSTM
                # next_state_embeds = torch.cat([
                #     state_encoder(action_text_embeddings[torch.tensor(ns)].unsqueeze(0))
                #     for ns in next_states
                # ], dim=0)

                #Mean pooling
                next_state_embeds, next_lengths = prepare_lstm_batch(next_states, action_text_embeddings)


                # Ensure shape of actions matches batch
                # action_embeds = action_text_embeddings[torch.tensor(actions)]
                # action_mask_tensor = torch.cat(action_mask, dim=0)
                # action_mask_n_tensor = torch.cat(action_mask_n, dim=0)
                #print(lengths_s)
                if isinstance(lengths_s, tuple):
                    lengths_s = list(lengths_s)
                lengths_s = torch.tensor(lengths_s, dtype=torch.int64, device="cpu")
                #q_values = q_network(state_embeds, lengths_s)
                q_values = q_network(state_embeds, action_mask, lengths_s)
                #next_q_values = target_q_network(next_state_embeds,lengths_ns)
                next_q_values = target_q_network(next_state_embeds, action_mask_n, lengths_ns)

                best_actions_from_policy = q_network(next_state_embeds,action_mask_n, lengths_ns).argmax(dim=1)
                next_q_values_best = next_q_values.gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)
                targets = rewards_tensor + gamma * next_q_values_best * (1 - dones_tensor)

                # Soft Q-Learning target:
                #temperature = 1
                #log_sum_exp = torch.logsumexp(next_q_values / temperature, dim=1)  # sum over actions
                #targets = rewards_tensor + gamma * temperature * log_sum_exp * (1 - dones_tensor)
                #targets = torch.clamp(targets, min=0.0, max=2.0)

                #loss = torch.nn.MSELoss()(q_values, targets.detach())
                actions = torch.tensor(actions)
                if len(actions)==1:
                    q_selected = q_values[0, actions.item()]
                else:
                     # shape: [4]
                    actions = actions.unsqueeze(1)
                    q_selected = q_values.gather(1, actions).squeeze(1) # shape: [4, 1]
                loss = F.smooth_l1_loss(q_selected, targets)
                #loss = F.smooth_l1_loss(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                #params = list(q_network.parameters()) + list(state_encoder.parameters())
                params = list(q_network.parameters())
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            # update target network
            if episode % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
            #per batch monitor
            monitor.log_batch(loss.item(), q_selected.mean().item(), targets.mean().item())


            if episode % 100 == 0:
                monitor.plot_all(episode)
                visualize_policy_and_q_values(env, q_network, episode, "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/PLOTS/plots_mask_retrain_physics/llmproxy_world_models_gpt_4o_mini/")
                visualize_state_embeddings_tsne(states, q_network,action_mask, action_text_embeddings,episode,"C:/Users/spaste01/PycharmProjects/Results/PPO_RL/PLOTS/plots_mask_retrain_physics/llmproxy_world_models_gpt_4o_mini/")
                save_checkpoint(q_network, optimizer, episode,
                                path="C:/Users/spaste01/PycharmProjects/Results/PPO_RL/CHECKPOINTS/check_points_gpt_4o_with_world_model/")

            if episode % 10 == 0:
                #save_checkpoint(q_network,optimizer, episode, path="C:/Users/spaste01/PycharmProjects/Results/PPO_RL/checkpoints")
                print("Targets:", targets)
                print("Q-values:", q_selected)
                print(f"Episode {episode}: Loss: {loss.item():.4f}, Buffer size: {len(replay_buffer)}, epsilon: {eps_threshold}")

    return q_network

# ----------------------- Run the setup -------------------------- #
env = RecipeEnv(bag_of_actions = steps, preconditions=preconditions)
load_checkpoint = True
path_checpoints = "C:/Users/spaste01/PycharmProjects/Results/PPO_RL/checkpoints_mask_cur"

# q_network = LSTMDQN(input_dim=768, hidden_dim=512, output_dim=len(actions_start))
# target_q_network = LSTMDQN(input_dim=768, hidden_dim=512, output_dim=len(actions_start))

q_network = LSTMDQN_Mask(input_dim=768, hidden_dim=512, num_actions=env.max_steps, output_dim=len(actions_start))
target_q_network = LSTMDQN_Mask(input_dim=768, hidden_dim=512, num_actions=env.max_steps,output_dim=len(actions_start))

target_q_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(list(q_network.parameters()), lr=1e-3)

if load_checkpoint:
    # Load checkpoint
    checkpoint = torch.load("C:/Users/spaste01/PycharmProjects/Results/PPO_RL/CHECKPOINTS/check_points_gpt_4o_with_world_model/checkpoint_ep1800.pth", map_location='cpu')  # or 'cuda' if using GPU
    q_network.load_state_dict(checkpoint['q_network'])
    target_q_network.load_state_dict(checkpoint['q_network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
q_network = dqn_training_loop_discounted(env, q_network, target_q_network, optimizer, action_text_embeddings, goal, goal_text_embedding, steps, metric, num_episodes=5000)


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

