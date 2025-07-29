import numpy as np
import random
from GoalBasedEnvironment import GoalBasedEnvironment
from RLAgent import RLAgent
import json
import yaml
import os
from utils.generate_results import PlotResults
from utils.add_labeled_sequence_to_json import extract_constrains

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, np.int64):
            return int(obj)  # Convert np.int64 to regular int
        return super().default(obj)


def main(config):
    json_dir = config['env']['json_path']
    save_dir = config['results']['save_dir']
    num_episodes = 100000
    train_flag = True
    if os.path.isdir(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                full_path = os.path.join(json_dir, filename)
            config_path = full_path
            with open(config_path, 'r') as file:
                data = json.load(file)
            # Path to your JSON file

            constraints = data.get("constraints_LLM", [])

            steps = data["steps"]
            num_actions = len(steps)
            num_states = num_actions
            # Assuming one-to-one mapping of steps to state ids

            # Create the environment
            env = GoalBasedEnvironment(env_config = config['env'],file_path = full_path)
            permutations_file = os.path.join(config['results']['save_backlog'],env.goal+'.json')

            #env_TD = GoalBasedEnvironment(env_config=config['env'], file_path=full_path)
            agent = RLAgent(agent_config = config['agent'],init_sequence_path = full_path,
                constraints=env.valid_transitions,
                state_space_size=num_states,
                action_space_size=num_actions,
                num_episodes=num_episodes, permutations_path=permutations_file
            )



            if train_flag:
                agent.train_LLM(env, full_path, num_episodes=num_episodes)
                #agent_TD.train_nSarsa(env_TD, num_episodes=num_episodes)
                if config['agent']['train']['mode']=='MCC':
                    res = {}
                    res['Q'] = agent.Q
                    res['target_policy'] = agent.target_policy
                    res['rewards_hist'] = agent.reward_hist
                    res['env_constrains'] = env.valid_transitions
                    res['res_constrains_updated'] = env.update_valid_transitions
                    res['goal'] = env.goal
                    res['steps'] = env.actions

                    file_name = 'MCC_LLM_' + env.goal + '.json'
                    with open(file_name, "w") as f:
                        json.dump(res, f, indent=4, cls=CustomEncoder)

                else:
                    res = {}
                    res['Q'] = agent_TD.Q
                    res['target_policy'] = agent_TD.target_policy
                    res['rewards_hist'] = agent_TD.reward_hist
                    res['env_constrains'] = env_TD.valid_transitions
                    res['res_constrains_updated'] = env_TD.update_valid_transitions
                    res['goal'] = env_TD.goal
                    res['steps'] = env_TD.actions

                    file_name = 'SARSA_n3_agent_' + env_TD.goal + '.json'
                    with open(file_name, "w") as f:
                        json.dump(res, f, indent=4, cls=CustomEncoder)


                # Print learned policy
                policy = agent.generate_target_policy()
                #policy = agent.target_policy
                print("\nLearned Policy:")
                state_u = 0
                for state, action in policy.items():
                    state = state_u
                    action = policy[state]
                    print(f"State {state} -> Action {action} ({steps[str(action)]})")
                    state_u = action
                    if state_u==env.end_state:
                        break

                #prepare for plot
                rewards = [agent.reward_hist]
                gen_res = PlotResults(env = env, Q = agent.Q, rewards = rewards, save_dir = save_dir)
                gen_res.plot_rewards()
            else:
                backlog_data = agent.prepare_backlog(env, full_path, num_episodes=num_episodes)
                constrains = extract_constrains(backlog_data['permutations_seq'])
                backlog_data['constraints_based_permutations']=constrains
                backlog_name = os.path.join(config['results']['save_backlog'], env.goal + '.json')
                with open(backlog_name, 'w', encoding='utf-8') as f:
                    json.dump(backlog_data, f, indent=2, ensure_ascii=False)
                y=1





if __name__ == "__main__":
    with open("POP_RL_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #file_path = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM\blenderbananapancakes.json"
    # config['env']['json_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"
    # config['agent']['init_sequence_path'] = r"C:\Users\Sveta\PycharmProjects\data\Cook\LLM"

    main(config)
