import json
import os
import uuid


class TrajectoryLogger:
    def __init__(self, json_path='experiment_log.json'):
        self.json_path = json_path
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def new_entry(self, trajectory, gt_label, reward_transition=None, explanation_transition = None, reward_contrastive = None,explanation_contrastive = None, reward_contrastive_cons=None,explanation_contrastive_cons=None,reward_cons_state = None, explanation_cons_state = None ,state_trace=None):
        entry_id = str(uuid.uuid4())
        return {
            entry_id: {
                'entry_id': entry_id,
                'sequence': trajectory,
                'state_trace': state_trace,
                'gt_label': gt_label,
                'reward_transition': reward_transition,
                'explanation_transition':explanation_transition,
                'reward_contrastive':reward_contrastive,
                'explanation_contrastive': explanation_contrastive,
                'reward_contrastive_cons':reward_contrastive_cons,
                'explanation_contrastive_cons':explanation_contrastive_cons,
                'reward_cons_state':reward_cons_state,
                'explanation_cons_state': explanation_cons_state
            }
        }, entry_id

    def update_direct_eval(self, entry, entry_id, reward, explanation):
        entry[entry_id]['llm_reward_direct'] = reward
        entry[entry_id]['llm_explanation_direct'] = explanation
        return entry

    def update_state_eval(self, entry, entry_id,state_trace, reward, explanation):
        entry[entry_id]['state_trace'] = state_trace
        entry[entry_id]['llm_reward_state_eval'] = reward
        entry[entry_id]['llm_explanation_state_eval'] = explanation
        return entry

    def log_entry(self, entry_id, entry):
        self.entries[entry_id] = entry

    def save(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=4)

    def get_entries(self):
        return self.entries
