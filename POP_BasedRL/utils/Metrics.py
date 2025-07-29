from POP_BasedRL.RLAgent import RLAgent
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Metrics:
    def __init__(self,q_table=None, true_constrains=None):
        self.q_table = q_table
        self.edges = true_constrains
    # === Metrics for Evaluating Learned Constraints ===
    def check_trajectory_edges(self,sequence):
        """
        Checks if the given sequence respects all directed edges in the GT graph.

        Args:
            sequence (list): List of action indices (e.g., [0, 5, 1, 7, ...])
            edges (list of tuples): List of directed edges (e.g., [(0, 1), (1, 7), ...])

        Returns:
            dict: {
                'respected_edges': list of edges that are respected,
                'violated_edges': list of edges that are violated,
                'all_respected': True if no violations
            }
        """
        index_map = {action: idx for idx, action in enumerate(sequence)}

        respected = []
        violated = []
        violated_seq = []
        for src, tgt in self.edges:
            if src in index_map and tgt in index_map:
                if index_map[src] < index_map[tgt]:
                    respected.append((src, tgt))
                else:
                    violated.append((src, tgt))
                    violated_seq.append((tgt,src))

        return {
            'respected_edges': respected,
            'violated_edges': violated,
            'violated_in_current_sequence': violated_seq,
            'all_respected': len(violated) == 0
        }

    def extract_learned_policy(q_table):
        """Extract greedy policy from Q-table."""
        policy = {}
        for state, actions in q_table.items():
            if actions:
                best_action = max(actions, key=actions.get)
                policy[state] = best_action
        return policy


    def calculate_metrics_to_sequence(self,gt,pred):

        precision = precision_score(gt, pred, zero_division=0)
        recall = recall_score(gt, pred, zero_division=0)
        f1 = f1_score(gt, pred, zero_division=0)

        gt_array = np.array(gt)
        pred_array = np.array(pred)

        TP = np.sum((pred_array == 1) & (gt_array == 1))
        TN = np.sum((pred_array == 0) & (gt_array == 0))
        FP= np.sum((pred_array == 1) & (gt_array == 0))
        FN = np.sum((pred_array == 0) & (gt_array == 1))

        accuracy = (TP + TN) / (TP + FP + FN + TN)

        return accuracy,f1,precision,recall

    def compare_to_true_constraints(self,learned_policy, learned_edges, true_edges):
        """Compute precision, recall, F1 against true constraints."""
        if learned_policy:
            learned_edges = set((s, a) for s, a in learned_policy.items())
            true_edges = set(self.edges)

        y_GT = np.array(true_edges).flatten()
        y_pred = np.array(learned_edges).flatten()

        # Only compute scores for the positive class (1s)
        precision = precision_score(y_GT, y_pred, zero_division=0)
        recall = recall_score(y_GT, y_pred, zero_division=0)
        f1 = f1_score(y_GT, y_pred, zero_division=0)

        TP = np.sum((y_pred == 1) & (y_GT == 1))
        FP = np.sum((y_pred == 1) & (y_GT == 0))
        FN = np.sum((y_pred == 0) & (y_GT == 1))

        # print("TP:", TP, "FP:", FP, "FN:", FN)
        #
        # precision = TP / (TP + FP + 1e-8)
        # recall = TP / (TP + FN + 1e-8)
        # f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "TP":TP,
            "FP":FP,
            "FN":FN
        }