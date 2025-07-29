import matplotlib.pyplot as plt
#from POP_BasedRL.RLAgent import RLAgent
#from POP_BasedRL.GoalBasedEnvironment import GoalBasedEnvironment
import numpy as np
import seaborn as sns
import os
import json
from POP_BasedRL.utils.Metrics import Metrics
from POP_BasedRL.GPTFeedbackConnector import GPTFeedbackConnector
from sklearn.metrics import precision_score, recall_score, f1_score
from POP_BasedRL.utils.collect_results import TrajectoryLogger

class PlotResults:
    def __init__(self, env, Q, rewards, save_dir, num_episodes=5000):
        self.env = env
        self.Q = Q
        #self.target_policy = self.generate_optimal_policy()
        self.rewards = rewards
        self.save_dir = save_dir
        self.window_size = min(100,num_episodes//100)

    def generate_optimal_policy(self):
        target_policy = {}
        for state in self.Q.keys():
            target_policy[state] = RLAgent.max_argmax(self,input=self.Q[state])
        return target_policy

    def moving_average(self,y,window_size):
        """Compute moving average for smoothing."""
        return np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    def generate_trajectories(self):
        trajectories = [[],[]]
        for i in range(2):
            self.env.reset()
            state = RLAgent.convert_state_to_key(self,state=(self.env.state,self.env.speed))
            finished = False
            while finished==False:
                trajectories[i].append(state)
                action = self.target_policy[state]
                next_state, next_speed, reward, finished = self.env.one_step(action = action)
                state = RLAgent.convert_state_to_key(self,state=(next_state,next_speed))
            trajectories[i].append(state)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(2):
            map = self.env.map.copy()
            for traj in trajectories[i]:
                map[traj[0], traj[1]] = 0.6
              # Create 1 row, 2 columns
            axes[i].imshow(map)
            sns.heatmap(map, linewidths=1, ax=axes[i])
            axes[i].set_title('trajectories trace B no acc env')

        plt.savefig(f'./plots/trajectories trace B no acc.png')  # Save the figure
        plt.show()

    def plot_rewards(self):
        y=[]
        save_file = self.save_dir+'/plots/MCC_batch_rewards_'+self.env.goal +'.png'
        title = 'Rewards -  MCC batch with seq init ' +self.env.goal

        # Apply moving average smoothing
        window_size = self.window_size
        if len(self.rewards)==2:
            for i in range(np.size(self.rewards,axis = 1)):
                y.append(self.rewards[i])
            y1 = np.array(self.rewards[0])
            y2 = self.rewards[1]
            x = np.arange(len(y1))
            # Adjust for better smoomthing
            y1_smooth = self.moving_average(y1, window_size)
            y2_smooth = self.moving_average(y2, window_size)
            x_smooth = x[:len(y1_smooth)]

        else:
            y = self.rewards[0]
            x = np.arange(len(y))
            if window_size==0:
                y1_smooth = y
                x_smooth = x
            else:
                if len(y)>window_size*20:
                    y1_smooth = self.moving_average(y, window_size)
                    x_smooth = x[:len(y1_smooth)]
                else:
                    y1_smooth = y
                    x_smooth = x
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_smooth, y1_smooth, 'b-o', label="epsilon=0.2", alpha=0.8)
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.legend()
            plt.grid(True)
            plt.ylim([-100, 100])
            plt.title(title)
            plt.savefig(save_file)
            plt.show()

    def metrics_plots(self,scores,goal_list, name):
        # Stats
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        save_dir = self.save_dir

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(goal_list, scores, color='skyblue')
        plt.xticks(rotation=90, fontsize=9)
        plt.ylabel("Score (e.g., F1)")
        plt.title("F1 Score per Goal "+name)

        # Mean line
        plt.axhline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_score:.2f}')
        # Shaded area for ±1 std
        plt.fill_between(range(len(scores)), mean_score - std_score, mean_score + std_score,
                         color='red', alpha=0.2, label=f'±1 Std = {std_score:.2f}')

        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom',
                     fontsize=8)

        title_name = save_dir+name+".png"
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(title_name, dpi=300)
        plt.show()

    def heatmaps_from_constarins(self, edges, goal,llm_flag=False):

        # Your edge list
        # edges = [
        #     [14, 2], [7, 3], [13, 2], [2, 4], [11, 6], [5, 7],
        #     [12, 7], [1, 7], [9, 7], [6, 8], [8, 10], [3, 11],
        #     [10, 14], [0, 1], [0, 5], [0, 9], [0, 12], [0, 13], [4, 15]
        # ]

        # Determine the number of nodes
        num_nodes = max(max(e) for e in edges) + 1  # max index + 1

        # Initialize adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Fill in the edges
        for src, tgt in edges:
            adj_matrix[src, tgt] = 1

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(adj_matrix, annot=True, cmap="Blues", cbar=False, square=True,
                    xticklabels=range(num_nodes), yticklabels=range(num_nodes))

        if llm_flag:
            title_name = "Adjacency Matrix Heatmap LLM based " + goal
        else:
            title_name = "Adjacency Matrix Heatmap GT "+goal
        plt.title(title_name)
        plt.xlabel("To Node")
        plt.ylabel("From Node")

        if llm_flag:
            save_path = self.save_dir + '_' + goal + '_adjacency_heatmap_LLM_based.png'
        else:
            save_path = self.save_dir+'_'+goal+'_adjacency_heatmap.png'
        # Save the heatmap
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the heatmap
        plt.show()

    def convert_edges_to_adj_matrix(self,edges,num_nodes):
        # Initialize adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for src, tgt in edges:
            adj_matrix[src, tgt] = 1
        return adj_matrix

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
    valid_sequence_2 = [0, 5, 1, 9, 12, 7, 3, 13, 11, 6, 8, 10, 14, 2, 4, 15]
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


def main(input_folder, load_folder, save_dir_plots, results_type, name):
    results_log_LLM = TrajectoryLogger(json_path="C:/Users/spaste01/PycharmProjects/Results/PPO_RL/LLM_evaluation_trajectories/blenderbananapancakes.json")
    load_good_episodes = False
    results_class = PlotResults([], [], [], save_dir_plots, num_episodes=5000)
    Metrics_class = Metrics([], [])
    f1_score_value = []
    goal_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)
        edges = data.get("edges", {})
        steps = data.get("steps", {})
        num_nodes = len(steps)
        goal = os.path.splitext(filename)[0]
        world_model = data.get("validate_constarins_llm_proxy_4o_mini", {})

        metrics = data.get("metrics_results_4", {})
        # 1. Convert keys to integers and gather all edges from preconditions
        edges_llm = []
        for target_str, data_w in world_model.items():
            target = int(target_str)
            for source_str in data_w["preconditions"]:
                source = int(source_str)
                edges_llm.append((source, target))


        for results_type_i in results_type:

            if results_type_i == 'metric_based_multiple_traj':
                filename = 'blenderbananapancakes.json'
                input_path = os.path.join(load_folder, filename)
                if os.path.exists(input_path):
                    with open(input_path, 'r') as file:
                        data_results_collection = json.load(file)
                    # fill the buffer with good examples
                if load_good_episodes:
                    metric = Metrics(true_constrains=edges)
                    metric_llm = Metrics(true_constrains=edges_llm)
                    permutations_path = 'C:/Users/spaste01/Documents/Research/data/backlog_data/blenderbananapancakes.json'
                    with open(permutations_path, 'r') as file:
                        data = json.load(file)
                        permutations_seq = data['permutations_seq']
                        for trajectory in permutations_seq:
                            reward,explanation = get_llm_feedback(trajectory, [], goal, steps, world_model, metric,metric_llm, results_log_LLM, [])

                gt_labels = []
                llm_reward_transition = []
                llm_reward_contrastive = []
                llm_reward_contrastive_cons = []
                llm_reward_cons_state = []
                idx = []
                evaluation_constrains = []
                llm_reward_cons_state_llm = []
                metric = Metrics(true_constrains=edges_llm)

                for result in data_results_collection:
                    info = data_results_collection[result][result]
                    gt_labels.append(info.get('gt_label'))
                    evaluation_constrains.append(int((metric.check_trajectory_edges(info.get('sequence')))['all_respected']))
                    llm_reward_transition.append(info.get('reward_transition'))
                    llm_reward_contrastive.append(info.get('reward_contrastive'))
                    llm_reward_contrastive_cons.append(info.get('reward_contrastive_cons'))
                    llm_reward_cons_state.append(info.get('reward_cons_state'))

                    idx.append((info.get('entry_id')))

                # Remove None values if any (optional)
                gt_labels = [p for p in gt_labels if p is not None]
                llm_reward_transition = [p for p in llm_reward_transition if p is not None]
                llm_reward_contrastive = [p for p in llm_reward_contrastive if p is not None]
                llm_reward_contrastive_cons = [p for p in llm_reward_contrastive_cons if p is not None]
                llm_reward_cons_state = [p for p in llm_reward_cons_state if p is not None]

                accuracy_transition,f1_transition,precision_transition,recall_transition = metric.calculate_metrics_to_sequence(gt_labels,llm_reward_transition)
                accuracy_cons, f1_cons, precision_cons, recall_cons = metric.calculate_metrics_to_sequence(
                    gt_labels, evaluation_constrains)
                accuracy_contrastive , f1_contrastive , precision_contrastive , recall_contrastive  = metric.calculate_metrics_to_sequence(gt_labels,llm_reward_contrastive)
                accuracy_contrastive_cons, f1_contrastive_cons, precision_contrastive_cons, recall_contrastive_cons = metric.calculate_metrics_to_sequence(gt_labels,
                                                                                                               llm_reward_contrastive_cons)
                # accuracy_contrastive_cons, f1_contrastive_cons, precision_contrastive_cons, recall_contrastive_cons = metric.calculate_metrics_to_sequence(evaluation_constrains,
                #                                                                                                pred_state)
                accuracy_cons_state, f1_cons_state, precision_cons_state, recall_state_cons_state = metric.calculate_metrics_to_sequence(gt_labels,llm_reward_cons_state)


            if results_type_i=='metrics_plots':
                f1_score_value.append(metrics.get("f1_score", 0))
                goal_list.append(goal)

            if results_type_i == 'heatmaps':
                results_class.heatmaps_from_constarins(edges_llm,goal,llm_flag=True)

            if results_type_i=='metrics':
                # # Initialize adjacency matrix
                adj_matrix_GT = results_class.convert_edges_to_adj_matrix(edges,num_nodes)
                adj_matrix_LLM = results_class.convert_edges_to_adj_matrix(edges_llm,num_nodes)

                metrics_results = Metrics_class.compare_to_true_constraints([],adj_matrix_LLM,adj_matrix_GT)

                # Convert NumPy types to native Python types
                metrics_serializable = {k: int(v) if isinstance(v, np.integer) else v for k, v in metrics_results.items()}

                # Convert to JSON
                data['metrics_results_llmproxy_gpt_4o_mini'] = metrics_serializable

                # Step 3: Save the updated file
                with open(input_path, "w") as f:
                    json.dump(data, f, indent=2)


    for results_type_i in results_type:
        if results_type_i=='metrics_plots':
            results_class.metrics_plots(f1_score_value, goal_list, name)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Run GPT constraint generation on JSON action files.")
    # parser.add_argument("folder", help="Path to the folder with JSON files")
    # parser.add_argument("--api_key", help="OpenAI API key (optional if set in environment)")
    # args = parser.parse_args()
    folder = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/"
    save_dir = r"C:/Users/spaste01/PycharmProjects/Results/PPO_RL/plots_mask_retrain_physics/Results_heatmaps/Metrics/"
    #load_dir contains the results of comparing traj to GT whether it follows the
    # constrains and the LLM evaluation with different prompts
    load_dir = r"C:/Users/spaste01/PycharmProjects/Results/PPO_RL/LLM_evaluation_trajectories"
    # main(input_folder=folder)
    main(input_folder=folder,load_folder = load_dir, save_dir_plots = save_dir,results_type = ['metric_based_multiple_traj'], name='F1_gpt_4')
    y=1