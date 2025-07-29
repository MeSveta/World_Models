# Tabular SARSA implementation for your GoalBasedEnvironment
import numpy as np
import gym
import json
from collections import defaultdict
import yaml
import os

# === Load your GoalBasedEnvironment ===
from GoalBasedEnvironment import GoalBasedEnvironment
from utils.generate_results import PlotResults
import matplotlib.pyplot as plt
from collections import Counter
import os
import yaml


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, np.int64):
            return int(obj)  # Convert np.int64 to regular int
        return super().default(obj)
# === SARSA Agent ===
class TabularSARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def max_argmax(self, input):
        best_action = np.random.choice([i for i, value in enumerate(input) if value == max(input)])
        return best_action

    def generate_target_policy(self):
        """Returns the greedy policy based on Q-values."""
        return {state: self.max_argmax(self.q_table[int(state)]) for state in range(self.n_actions-1)}


    def print_policy(self):
        print("\nFinal policy (greedy actions):")
        for state in sorted(self.q_table.keys()):
            print(f"State {state}: Action {np.argmax(self.q_table[state])} | Q-values: {self.q_table[state]}")

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.max_argmax(self.q_table[state])

    def update(self, s, a, r, s_, a_):
        target = r if a_ == 0 and r <= 0 else r + self.gamma * self.q_table[s_][a_]
        self.q_table[s][a] += self.alpha * (target - self.q_table[s][a])

# === RL-Compatible step function ===
def rl_step(env, action):
    reward = 0.0
    done = False
    info = {}

    env.steps_taken += 1

    if action in env.visited_actions:
        reward = -1.0
        done = True
        info["reason"] = "repeated action"

    elif env.steps_taken >= env.max_steps:
        reward = -5.0
        done = True
        info["reason"] = "step limit reached"

    elif action == env.end_state and len(env.visited_actions) < len(env.actions) - 1:
        reward = -10.0
        done = True
        info["reason"] = "premature END"

    elif action == env.end_state and len(env.visited_actions) == len(env.actions) - 1:
        reward = 10.0
        done = True
        info["reason"] = "successful completion"

    elif not env.state == env.end_state:
        if action not in env.update_valid_transitions.get(env.state, []):
            reward = -5.0
            done = True
            info["reason"] = "invalid transition"
        else:
            required = env.preconditions.get(action, set())
            if not required.issubset(env.visited_actions):
                reward = -5.0
                done = True
                info["reason"] = f"violated preconditions for action {action}"
                info["missing"] = list(required - set(env.visited_actions))
            else:
                reward = 6.0
                info["reason"] = "valid logical transition"

    env.state = action
    env.current_step = action
    env.visited_actions.append(action)

    return env.state, reward, done, info


# === Training Loop ===
def train_sarsa(env, json_path, num_episodes=5000):
    rewards = []
    reason_log = []
    success_count = 0



    n_states = len(env.actions)
    n_actions = env.action_space.n

    agent = TabularSARSAAgent(n_states, n_actions)

    for ep in range(num_episodes):
        agent.epsilon = max(0.01, agent.epsilon * 0.9)
        episode_trace = []
        state = env.reset()
        #env.visited_actions.append(0)
        action = agent.get_action(state)
        ep_reward = 0
        done = False

        while not done:
            next_state, reward, done, info = rl_step(env, action)
            next_action = agent.get_action(next_state)
            # Skip update if repeated action to avoid learning it as good
            if not done or (info.get("reason") != "repeated action" and reward > 0):
                agent.update(state, action, reward, next_state, next_action)
            if done: #and not (info.get("reason") != "repeated action" and reward > 0):
                agent.update(state, action, reward, next_state, 0)

            # #state, action = next_state, next_action
            # if done:
            #
            # else:
            #     agent.update(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            ep_reward += reward
            episode_trace.append((state, action, reward, info))

        rewards.append(ep_reward)
        episode_reasons = [info.get("reason", "") for (_, _, _, info) in episode_trace if isinstance(info, dict)]
        reason_log.extend(episode_reasons)

        if "successful completion" in episode_reasons:
            success_count += 1

        print(f"Episode {ep} | Reward: {ep_reward:.2f} | Trace: {episode_trace} | Reasons: {episode_reasons}")

    agent.print_policy()
    return env, agent, rewards


# === Visualization ===
def plot_metrics(rewards, reason_log, window=100):
    moving_avg = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]
    reason_counter = Counter(reason_log)
    success_rate = [reason_log[:i + 1].count("successful completion") / (i + 1) for i in range(len(rewards))]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    plt.plot(moving_avg, label=f'{window}-ep Moving Avg')
    plt.plot(success_rate, label='Success Rate')
    plt.title('Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward / Success')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(reason_counter.keys(), reason_counter.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Episode Termination Reasons')
    plt.tight_layout()
    plt.show()

# === Example usage ===
if __name__ == "__main__":
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
            # with open(json_path, 'r') as file:
            #     data = json.load(file)
            # # Path to your JSON file

    #json_path = "C:/Users/spaste01/Documents/Research/data/blenderbananapancakes.json"  # replace with actual path
            env = GoalBasedEnvironment(env_config, json_path)
            agent, reward_log = train_sarsa(env, json_path)
            rewards = [reward_log]
            gen_res = PlotResults(env=env, Q=agent.q_table, rewards=rewards, save_dir=config["results"]["save_dir"])


            # Saving and plotting results
            res = {}

            # Print learned policy
            policy = agent.generate_target_policy()
            policy_seq = {}
            steps = env.actions
            print("\nLearned Policy:")
            state_u = 0
            for ii,(state, action) in enumerate(policy.items()):
                state = state_u
                action = policy[state]
                print(f"State {state} -> Action {action} ({steps[str(action)]})")
                policy_seq[ii] = f"State {state} -> Action {action} ({steps[str(action)]})"
                state_u = action
                if state_u == env.end_state:
                    break


            res['Q'] = {int(k): list(v) for k, v in agent.q_table.items()}
            res['rewards_hist'] = reward_log
            res['env_constrains'] = []
            res['res_constrains_updated'] = env.update_valid_transitions
            res['goal'] = env.goal
            res['steps'] = env.actions
            res['target_policy'] = policy
            res['target_policy_sequence'] = policy_seq

            file_name = 'Sarsa_' + env.goal + '.json'
            with open(file_name, "w") as f:
                json.dump(res, f, indent=4, cls=CustomEncoder)

            gen_res.plot_rewards()



y=1
