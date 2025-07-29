import openai
import os
import sys
import json
from openai import OpenAI
from POP_BasedRL.LLMProxy.llmproxy import generate
import re
from POP_BasedRL.utils.Metrics import Metrics

OPENAI_API_KEY = ""


def sanitize_json_string(s):
    # Remove all control characters except \t, \n, and \r
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

class GPTFeedbackConnector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key


    def generate_constraints_dot(self, actions, goal):
        """
        Given a bag of actions and a goal, use GPT to generate initial constraints and explanation.
        Return a dict: {"constraints": [...], "explanation": "..."}
        Includes validation and retries.
        """
        prompt = (
            f"Given the following goal: '{goal}', and the following actions:\n"
            f"{actions}\n"
            "Return the logical or temporal dependencies between these actions in a dot language as a list of [a, b] edges in the field \"constraints\",\n"
            "and provide a short explanation of the reasoning in the field \"explanation\".\n"
            "Return ONLY a JSON-compatible response like:\n"
            "{\"constraints\": [[0, 1], [1, 2], ...], \"explanation\": \"...\"}"
        )

        max_attempts = 3
        attempts = 0

        response = self._query_gpt(prompt)
        try:
            parsed = json.loads(response)
            cleaned_constraints = self.filter_constraints(parsed.get("constraints", []), actions)
            if self.validate_constraints(cleaned_constraints, actions):
                parsed["constraints"] = cleaned_constraints
                return parsed
        except Exception:
            pass


        return {
            "constraints": [],
            "explanation": "Invalid constraints format or action values after multiple attempts."
        }


    def llm_constrains(self, actions, goal,physics):
        prompt1 = f"""
        You are an AI system that analyzes action sequences to extract procedural and physical knowledge. 

            Your input is:
            - A list of actions (unordered), each associated with an index
            - A task goal (what the agent wants to achieve)
            - The physical knowledge of every action
            
            Your job is to extract a symbolic world model that maps each action to:
            - Effects (as descriptive strings)
            - Constraints (as natural-language constraints)
            - Whether the action is physically irreversible, does it effect is irreversible
            - Preconditions (as indices of other actions that must occur before, including START - '0' and END-'15'), based on the effect and the constrains
            Input actions:
            {actions}
            
            Physical knowledge:
            {physics}
        
            Goal: {goal}
            
            Output a JSON dictionary in this format:
            
            {{
              "0": {{
                "preconditions": [],
                "effects": [...],
                "constraints": [...],
                "irreversible": false
              }},
              ...
            }}
            
            Rules:
            - Use effects and constraints in descriptive text.
            - Be physics-aware 
            - Add constraints when certain actions must happen before/after others for physical or logical reasons.
            - Preconditions must be action indices.
            - Use action numbers, not names, for preconditions.
            
            Now output the structured world model for this action set.

        """
        parsed_barch = []
        #
        # for i in range(l_action_sequence//batch_LLM+1):
        #     prompt = prompt1
        #     actions_text = []
        #     for seq in action_sequence[i*batch_LLM:min(l_action_sequence,(i+1)*batch_LLM)]:
        #         actions_text = [actions[str(i)] for i in seq]
        #         prompt += f"- {actions_text}\n\n"
        #         if actions_text==[]:
        #             break
        #     if actions_text == []:
        #
        #         break

        prompt = prompt1
        #raw_response = self._query_gpt(prompt)
        raw_response = self.query_llm_proxy(prompt)
        raw_response_json = raw_response['response']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response_json.startswith("```json"):
                raw_response = raw_response_json.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)
        return parsed

    def validate_constrains(self, actions, goal, physics,constraints):
        prompt1 = f"""
        You are an AI system that analyzes action sequences to extract procedural and physical knowledge. 

            Your input is:
            - A list of actions (unordered), each associated with an index
            - A task goal (what the agent wants to achieve)
            - The physical knowledge of every action
            - A list of constrains

            Your job is to modify the list of preconditions based on all the constrains and effects you got.
            - Effects (as descriptive strings)
            - Constraints (as natural-language constraints)
            - Whether the action is physically irreversible, does it effect is irreversible
            - Preconditions (as indices of other actions that must occur before, including START and END), based on the effect and the constrains
            Input actions:
            {actions}

            Physical knowledge:
            {physics}

            Goal: {goal}
            
            Constrains:{constraints}

            Output a JSON dictionary in this format:

            {{
              "0": {{
                "preconditions": [],
                "effects": [...],
                "constraints": [...],
                "irreversible": false
              }},
              ...
            }}

            Rules:
            - Use effects and constraints in descriptive text.
            - Be physics-aware 
            - Add constraints when certain actions must happen before/after others for physical or logical reasons.
            - Preconditions must be action indices.
            - Use action numbers, not names, for preconditions.

            Now output the structured world model for this action set.

        """
        parsed_barch = []

        prompt = prompt1
        #raw_response = self._query_gpt(prompt)
        raw_response = self.query_llm_proxy(prompt)
        raw_response_json = raw_response['response']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response_json.startswith("```json"):
                raw_response = raw_response_json.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)
        return parsed

    def generate_pddl(self, actions, goal, world_model):
        prompt1 = f"""
        You are an expert in AI planning and PDDL. I will give you a world model that includes a set of actions with their descriptions, constraints, and preconditions. Your task is to generate a **complete and valid PDDL domain file**, following these requirements:

        1. domain.pddl file that:
            1. Use `(:requirements :strips)` and any other needed requirements.
            2. Define all relevant `:types` and `:predicates`, based on the world model.
            3. For each action:
               - Include `:parameters`, even if empty.
               - Translate constraints into `:preconditions`.
               - Translate effects into `:effects`.
            4. Make sure predicates are defined consistently and used in both preconditions and effects.
            5. Name predicates logically (e.g., `banana-added`, `butter-melted`, etc.).
            6. Ensure there are **no type or name conflicts**, and the domain is parsable.
            7. Follow proper PDDL syntax exactly.

        
        2. A `problem.pddl` file that:
           - Includes the object definitions (e.g., egg, banana, blender, pancake)
           - Defines the initial state (assume no steps have occurred)
           - Sets the goal to complete step 15: "End the cooking process"
        
        Now here is the world model and action list (first action is 0, second is 1, etc.):{world_model}
        
        - the mapping of action numbers to names {actions}
        - goal: {goal}
        
        Now output a valid JSON dictionary with the following structure (no markdown fences, no explanation):
        ''' json
        {{
          "domain": "<full domain.pddl string here>",
          "problem": "<full problem.pddl string here>"
        }}
        
        The values must be escaped properly so the JSON can be parsed with `json.loads()`.
        """


        parsed_barch = []

        prompt = prompt1
        # raw_response = self._query_gpt(prompt)
        raw_response = self.query_llm_proxy(prompt)
        raw_response_json = raw_response['response']

        # Attempt to sanitize and parse the response
        # try:
        #     # Sometimes GPT wraps the response in a code block, strip it
        #     if raw_response_json.startswith("```json"):
        #         raw_response = raw_response_json.strip("```json").strip("```").strip()

        # except json.JSONDecodeError as e:
        #     print("❌ Failed to parse GPT response as JSON. Response was:")
        #     print(raw_response)
        #     return {
        #         "reward": 0,
        #         "transitions": [],
        #         "explanation": "Failed to parse GPT output"
        #     }

        #clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(raw_response_json)
        return parsed
    
    def trajectory_action_explanation(self,actions,goal,action_sequence):
        prompt1 = f"""
            You are an expert in AI planning and plan evaluation. I will give you a **goal** and a sequence of actions that aim to achieve this goal. Your task is to explain the **purpose of each action**, in context of the goal.

            Please return a JSON dictionary in the following structure (no markdown, no explanations outside the JSON):

            {{
            "goal": "{goal}",
            "trajectory": [
                {{"action": "{actions[str(action_sequence[0])]}", "explanation": "..."}},
                {{"action": "{actions[str(action_sequence[1])]}", "explanation": "..."}},
                ...
            ]
            }}

            Each explanation should describe why the action is useful or necessary toward achieving the goal.
            Here is the goal: {goal}
            Here is the sequence of actions:
            {action_sequence}
            """

        for seq in action_sequence:
            actions_text = [[actions[str(i[0])],actions[str(i[1])]] for i in seq]
            prompt =prompt1 + f"- {actions_text}\n"

        raw_response = self.query_llm_proxy(prompt)
        raw_response_json = raw_response['response']





    def clean_json_like_string(self,raw):
        import re

        if not raw or not isinstance(raw, str):
            return ""

        raw = raw.strip()

        # Remove markdown block
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)  # remove ```json or ```
            raw = re.sub(r"\s*```$", "", raw)  # remove ending ```

        return raw.strip()

    def evaluate_contrastive(self, action_sequence, actions, goal, valid_sequence_1, valid_sequence_2,invalid_sequence_1, invalid_sequence_2):
        prompt = f"""
        You are a reasoning expert evaluating whether a candidate sequence of actions logically and causally achieves a given goal.
        ### GOAL:
        "{goal}"
        You will be shown:
        - A full list of possible actions (unordered)
        - Two known **valid sequences** that achieve the goal
        - Two known **invalid sequences** that fail due to ordering or logic errors
        - A **candidate sequence** to evaluate

        Each sequence is a list of action indices. The same goal can be achieved in multiple ways — your task is to judge whether the candidate plan is a valid structure.
        ### Allen's Temporal Logic (important):
        Use Allen's Interval Algebra to reason about **temporal constraints**:
        - If action B must precede A, then the **interval of B must end before A starts**
        - If the sequence violates this temporal dependency (e.g., pouring before blending), it is incorrect

        ### Full List of Actions (indexed):
        {actions}

        ### Valid Examples (reference plans):
        
        #### Valid Plan 1:
        {valid_sequence_1}
        #### Valid Plan 2 (alternate valid structure):
        {valid_sequence_2}
        These both succeed logically. Their ordering respects causal and temporal dependencies necessary to achieve the goal.
        ### Invalid Examples:
        #### Invalid Plan 1:
        {invalid_sequence_1}
        #### Invalid Plan 2:
        {invalid_sequence_2}
        ### Candidate Plan to Evaluate:
        {action_sequence}
        ### Instructions:
        1. Compare the candidate to both valid and invalid plans.
        2. Use **temporal reasoning** (e.g., Allen’s “before” relation) and **state feasibility** to judge whether the candidate respects physical/logical structure.
        3. If the candidate sequence resembles the valid examples in structure and effect, assign `"reward": 1`.
        4. If it contains flaws similar to the invalid examples, assign `"reward": 0`.
        5. Justify your answer — explain any violations or confirm valid transitions.
 
        ### Output Format (valid JSON):
        {{
          "reward": 1 or 0,
          "explanation": "Your reasoning, including key transitions, violations (if any), and comparison to examples."
        }}
        """

        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)

        # Make sure raw_response is a dict
        if isinstance(raw_response, dict) and 'response' in raw_response:
            raw_response = raw_response['response']
        else:
            print("⚠️ raw_response is not a dictionary or missing 'response' key.")
            parsed = {'reward': None,
                      'explanation': []}
            return parsed['reward'], parsed['explanation']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            parsed = {'reward': None,
                      'explanation': []}
            return parsed['reward'], parsed['explanation']

        clean_response = self.clean_json_like_string(raw_response)

        try:
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw response was:")
            print(clean_response)
            parsed = {'reward':None,
                      'explanation':[]}

        return parsed['reward'],parsed['explanation']

    def evaluate_contrastive_violation_constrains(self, action_sequence, actions, goal, violated_subsequences, valid_sequence_1, valid_sequence_2,
                             invalid_sequence_1, invalid_sequence_2):
        prompt = f"""
        You are a reasoning expert evaluating whether a candidate sequence of actions logically and causally achieves a given goal.

        ### GOAL:
        "{goal}"

        ### You are provided with:
        1. A **full list of possible actions** (unordered):
        {actions}

        2. Two **valid reference sequences** that successfully achieve the goal:
        - These demonstrate correct temporal and causal structure.

        #### Valid Plan 1:
        {valid_sequence_1}

        #### Valid Plan 2 (alternate form):
        {valid_sequence_2}

        3. Two **invalid reference sequences** that fail due to poor ordering, missing steps, or logical violations:

        #### Invalid Plan 1:
        {invalid_sequence_1}

        #### Invalid Plan 2:
        {invalid_sequence_2}

        4. A **candidate plan** (to evaluate):
        {action_sequence}

        5. A set of **violating subsequences**, extracted from known (but noisy) constraints:
        Each pair [A, B] means action A occurred before action B in the sequence — but may violate a known rule that **B should precede A**.

        This is a potential contradiction of **Allen's interval relation: 'B BEFORE A'**, but the plan contains **A ➜ B** instead.

        Violations observed in this candidate:
        {violated_subsequences}

        ### Allen's Temporal Logic:
        Use Allen’s Interval Algebra for reasoning about ordering:
        - If constraint says “B precedes A”, then **B must finish before A starts**
        - If A precedes B in the candidate, this may indicate a violation

        ### Instructions:

        1. **Compare** the candidate plan to the valid and invalid reference plans:
           - Does its **structure**, **ordering**, and **causal flow** resemble the valid ones?

        2. **Analyze** the violating subsequences:
           - Are they **true violations** that disrupt the goal?
           - Or are they false positives that don’t affect feasibility?

        3. Use **temporal reasoning**, **causal dependencies**, and **state logic** to justify your judgment.

        4. Your final judgment:
           - If the plan is coherent and the goal is still logically achievable, return `"reward": 1`
           - If key ordering problems exist (whether from the violations or others you spot), return `"reward": 0`

        ### Output Format (valid JSON):
        {{
          "reward": 1 or 0,
          "explanation": "Explain your decision, referring to violations, comparisons to reference plans, and any causal or temporal errors."
        }}
        """

        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)

        # Make sure raw_response is a dict
        if isinstance(raw_response, dict) and 'response' in raw_response:
            raw_response = raw_response['response']
        else:
            print("⚠️ raw_response is not a dictionary or missing 'response' key.")
            parsed = {'reward': None,
                      'explanation': []}
            return parsed['reward'], parsed['explanation']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            parsed = {'reward': None,
                      'explanation': []}
            return parsed['reward'], parsed['explanation']

        clean_response = self.clean_json_like_string(raw_response)

        try:
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw response was:")
            print(clean_response)
            parsed = {'reward':None,
                      'explanation':[]}
        return parsed['reward'], parsed['explanation']

    def evaluate_transition(self, action_sequence, actions, goal):
        prompt = f"""
       
        You are an expert in procedural reasoning and temporal planning.
        
        Task:
        Evaluate whether each proposed sequence of actions represents a temporally valid plan to achieve the following goal:
        
        Goal:
        "{goal}"
        
        You are given a *bag of actions* — an unordered list of all relevant actions:
        {actions}
        
        Each sequence is an ordered list of action transitions (e.g., ["Add flour" ➜ "Mix ingredients"]). You must evaluate if the temporal relations between actions are logically sound based on real-world constraints.
        
        Specifically:
        - Use the concept of temporal ordering from **Allen’s Interval Algebra**.
        - Consider whether each action A **precedes** (i.e., `before`) its successor B in a way that respects causal and goal-directed structure.
        
        Instructions:
        - Return `"reward": 1` if the sequence reflects a coherent partial order that could lead to the goal.
        - Return `"reward": 0` if any actions appear out of order, violate dependencies, or omit necessary steps.
        - Provide an `"explanation"` describing violations, using terms like **"X must precede Y"** or **"missing prerequisite for Y"**.
        
        ⚠ Return only valid JSON, exactly in this format:
        {{
          "sequence_scores": [
            {{
              "reward": 1 or 0
              "explanation": "..."          
            }},
            ...
          ]
        }}
        
        Here are the sequences to evaluate:
        {action_sequence}
        """

        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)
        # Unwrap if wrapped in a dict
        if isinstance(raw_response, dict):
            raw_response = raw_response.get("response", "")
        if not isinstance(raw_response, str):
            print("⚠️ raw_response is not a string.")
            return None

        # Remove code block markers if present
        if raw_response.strip().startswith("```json"):
            raw_response = raw_response.strip().strip("```json").strip("```").strip()

        # Try to sanitize and parse JSON
        try:
            clean_response = sanitize_json_string(raw_response)
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Original response:", raw_response)
            return {
                "reward": None,
                "explanation": "Failed to parse GPT output"
            }

        # Extract reward and explanation
        if isinstance(parsed, dict) and "sequence_scores" in parsed:
            seq_score = parsed["sequence_scores"]
            if isinstance(seq_score, list) and len(seq_score) > 0:
                reward = seq_score[0].get("reward", None)
                explanation = seq_score[0].get("explanation", None)
            else:
                reward = None
                explanation = "Empty sequence_scores list"
        else:
            reward = None
            explanation = "Missing 'sequence_scores' in parsed response"

        return reward,explanation


    def evaluate_transition_and_effects(self, action_sequence, actions, goal, effects):
        prompt = f"""

 
            You are an expert in temporal planning and physical process modeling.
            
            Task:
            Evaluate whether each proposed action sequence is temporally and causally valid for achieving the following goal:
            
            Goal:
            "{goal}"
            
            You are given:
            - A *bag of actions* (unordered): {actions}
            Each sequence is an ordered list of transitions (e.g., ["Add flour" ➜ "Mix ingredients"]).
            
            Use two criteria:
            1. **Temporal validity**: Check that each action A *precedes* (i.e., ends before) action B in a way consistent with Allen's interval algebra and real-world task logic.
            2. **Causal validity**: Ensure the physical effects of earlier actions enable or support the conditions for later actions. A sequence is invalid if key state transitions are missing or wrongly ordered.
            
            Instructions:
            - Return `"reward": 1` if the sequence reflects a temporally and causally coherent plan to achieve the goal.
            - Return `"reward": 0` otherwise.
            - Provide an `"explanation"` noting issues like:
              - "Action X must precede Y to enable required conditions"
              - "Effect of action A is missing before B"
              - "Sequence omits physical transformation needed for goal"
            
        ⚠ Return only valid JSON, exactly in this format:
        {{
          "sequence_scores": [
            {{
              "sequence": [ [from, to], [from, to], ... ],
              "reward": 1 or 0
              "explanation": "..."          
            }},
            ...
          ]
        }}

        Here are the sequences to evaluate:
        {action_sequence}
        """

        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)

        if isinstance(raw_response, dict):
            raw_response = raw_response.get("response", "")
        if not isinstance(raw_response, str):
            print("⚠️ raw_response is not a string.")
            return None
        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "sequence": [],
                "reward": 0,
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed_barch = []

        try:
            clean_response = sanitize_json_string(clean_response)
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            parsed = None

        if parsed is not None and 'sequence_scores' in parsed:
            parsed_barch.append(parsed['sequence_scores'])
            reward = parsed_barch[0][0].get('reward', None)
            explanation = parsed_barch[0][0].get('explanation', None)
        else:
            print("⚠️ Skipping due to missing or malformed data.")
            reward = None
            explanation = "Parsing failed or 'sequence_scores' missing"

        return reward, explanation

    def evaluate_batch(self, action_sequence, actions, goal):
        """
        Evaluate a final sequence of actions for chronological reasonableness.
        Return: {reward: 1 or 0, transitions: [problematic transitions] in dot format}
        """

        """
         Evaluate a final sequence of actions for chronological reasonableness.
         Return: {reward: 1 or 0, transitions: [...], explanation: "..."}
         """
        #actions_text = [str(i)+'-' + actions[str(i)] for i in action_sequence]
        #actions_text = [str(i) for i in action_sequence]
        prompt1 = (f"""
        You are an expert at recepies and in planning. Pat attention carefully for every action. Given the following goal: '{goal}', and the following bag of actions:
        {actions}
        
        Below are multiple sequences of actions. Each sequence should be evaluated for whether it reasonably and logically leads to achieving the goal,
        considering the required ordering of actions, dependencies, and logical transitions based on typical task structure.
        If a sequence makes logical chronological sense to achieve the goal, return 1. Otherwise, return 0. No redundancy or repetition allowed.
        
        🔧 For every sequence return the result in valid JSON format, using double quotes (") only. Do not use single quotes (') around keys or values.
        
        Respond exactly in the following JSON format:
        {{
          "sequence_scores": [
            {{"sequence": [list of steps], "reward": 0 or 1, "explanation":}}
            ... 
          ]
        }}
        
        Here are the sequences to evaluate:
        """)

        # l_action_sequence = len(action_sequence)
        # batch_LLM = min(5,l_action_sequence)
        parsed_barch = []
        #
        # for i in range(l_action_sequence//batch_LLM+1):
        #     prompt = prompt1
        #     actions_text = []
        #     for seq in action_sequence[i*batch_LLM:min(l_action_sequence,(i+1)*batch_LLM)]:
        #         actions_text = [actions[str(i)] for i in seq]
        #         prompt += f"- {actions_text}\n\n"
        #         if actions_text==[]:
        #             break
        #     if actions_text == []:
        #
        #         break
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
                # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
                }

        clean_response = self.clean_json_like_string(raw_response)


        parsed = json.loads(clean_response)

        parsed_barch.append(parsed['sequence_scores'])
            #parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
            # parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
            # print("bad transitions")
            # print(parsed['bad transitions'])
            # print(f"Explanation: {parsed['explanation']}\n")
            # print(f"reward: {parsed['reward']}\n")

        return parsed_barch[0][0]['reward']

    def generate_sequence_based_explanation(self, action_sequence, actions, goal, explanation_w):
        prompt1 = f"""
           You are a critical evaluator with deep knowledge of real-world processes, procedures, and planning.

           You msut correct the given sequence if it diverges from the given explanation
           
           ### GOAL:
           "{goal}"
           ### Bag of available actions (indexed):
           {actions}

           Use explanation and give actions to correct if needed the action_sequence
           explanation:{explanation_w}
           
           Output format:
           Respond strictly in this JSON format, using double quotes only:
          
               {{
                 "correct_sequence": [list of integers],
                 "explanation": "..."
                 "need_to_change": 1/0  '0' for need to change the sequence and '1' - if no changes needed 
               }}
         
           Here are the sequences to evaluate:
           [
           {action_sequence}
           ]
           """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)
        raw_response = raw_response['response']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "correct_sequence": 0,

                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)
        parsed = json.loads(clean_response)
        return parsed['correct_sequence'], parsed['explanation'], parsed['need_to_change']
    def evaluate_critic_with_world_model(self, action_sequence, actions, goal,world_model):
        prompt1 = f"""
        You are a critical evaluator with deep knowledge of real-world processes, procedures, and planning.

        You must determine whether a proposed sequence of actions is logically valid and physically feasible to reach the given goal and the knowledge of world model. 

        ---

        ### GOAL:
        "{goal}"
        ### Bag of available actions (indexed):
        {actions}

        ### Physical effects of each action (physics_data):
        Use world model to reason about what each action physically changes and whether the effect is irreversible:

        world_model:
        {world_model}

        ### Instructions:

        You will receive one or more sequences of action indices. For each sequence:

        1. Simulate the actions step-by-step as if performing them in the real world.
        2. At each step, retrieve the **physics** of the current action (Effect and Irreversible).
        3. Keep track of the **material state** 
        4. Before applying an action, check:
           - Is the **required physical state** present?
           - Has an **irreversible transformation** already occurred that makes the action invalid?
           - Does this step make sense at this time based on prior steps?
        5. Only give a reward of 1 if the sequence is:
           - Logically coherent
           - Physically valid according to `physics_data`
           - Free from contradictions or premature irreversible steps
        6. The explanation should relate to every action and explain why it is logical
        7. Constraints and the preconditions might be wrong and not full , aplay your own judgment
        Output format:
        Respond strictly in this JSON format, using double quotes only:
        {{
          "sequence_scores": [
            {{
              "sequence": [list of integers],
              "reward": 1 or 0,
              "explanation": "..."
            }}
          ]
        }}

        Here are the sequences to evaluate:
        [
        {action_sequence}
        ]
        """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)
        raw_response = raw_response['response']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed_barch = []

        try:
            clean_response = sanitize_json_string(clean_response)
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            parsed = None

        if parsed is not None and 'sequence_scores' in parsed:
            parsed_barch.append(parsed['sequence_scores'])
            reward = parsed_barch[0][0].get('reward', None)
            explanation = parsed_barch[0][0].get('explanation', None)
        else:
            print("⚠️ Skipping due to missing or malformed data.")
            reward = None
            explanation = "Parsing failed or 'sequence_scores' missing"

        return reward, explanation

    def evaluation_based_states(self,action_sequence, actions, goal, state_trace, explanation_w):
        prompt1 = f"""
        You are an expert in logical planning and process validation.

        Given a goal and a sequence of simulated states (produced step-by-step after each action), evaluate whether the full sequence is:
        - Logically coherent
        - Physically valid
        - Goal-achieving
        ---
        ### GOAL:
        "{goal}"
        ### Actions (indexed):
        {actions}
        ---
        explanation:
        {explanation_w}
        ### Instructions:
        1. Review the sequence of actions and associated `state_after` values.
        2. At each step, check if the change in state is consistent with the action's expected effects and preconditions.
        3. Identify the explanation given and the states are not contradicting.
        4. Determine whether the final state satisfies the goal.
        5. Provide a reward:
           - `reward = 1` if the sequence is valid, logically coherent, and achieves the goal.
           - `reward = 0` otherwise, if there is contradiction between te states and the explanation.
        6. Give a detailed explanation of why the score was given, referencing key steps.
        ---
        ### Input:
        action_sequence: {action_sequence}
        state_trace: {state_trace}  
        ---
        ### Output Format:
        {{
          "sequence_evaluation": {{
            "reward": 1 or 0,
            "explanation": "..."
          }}
        }}
        """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)
        parsed_barch = []

        # Make sure raw_response is a dict
        if isinstance(raw_response, dict) and 'response' in raw_response:
            raw_response = raw_response['response']
        else:
            print("⚠️ raw_response is not a dictionary or missing 'response' key.")
            return {
                "sequence_evaluation": 0,
                "reward": None,
                "explanation": "raw_response was not in expected format"
            }

        try:
            # Strip code block wrapper if present
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

            clean_response = self.clean_json_like_string(raw_response)
            parsed = json.loads(clean_response)

            if 'sequence_evaluation' not in parsed:
                raise KeyError("Missing 'sequence_evaluation' in parsed JSON")

            parsed_barch.append(parsed['sequence_evaluation'])

            reward = parsed_barch[0].get('reward', None)
            explanation = parsed_barch[0].get('explanation', "No explanation provided")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"❌ Error while parsing GPT response: {e}")
            print(f"🧪 Raw response was:\n{repr(raw_response)}")
            return {
                "sequence_evaluation": 0,
                "reward": None,
                "explanation": f"Parsing failed: {str(e)}"
            }

        return reward, explanation

    def state_evaluation(self,action_sequence, actions, goal,world_model, explanation_w):
        prompt1 = f"""
        You are an expert in logical planning and world modeling.
        Given a goal, a set of indexed actions, and a world model with their effects and constraints, simulate the progression of world state step-by-step for a given sequence of actions.

        ### GOAL:
        "{goal}"
        ### Actions (indexed):
        {actions}
    
        ### World Model (per action):
        Each action includes:
        - `preconditions`: required state before applying the action
        - `effects`: changes made to the world
        - `constraints`: any temporal or contextual requirements
        - `irreversible`: whether the action cannot be undone
        
        {world_model}
        --
        ### CRITICAL RULE (do not skip this):
        **If an action's preconditions are not satisfied at the time it is encountered, set `"state_after": "not applicable"` and do NOT apply its effects. Continue the trace. Do not pretend the action happened.**
        This rule is mandatory. All preconditions must be enforced strictly.
        ---
        ### Instructions:
        1. Initialize an empty world state. You may create variables like:
           - `blender_contents`, `butter_melted`, `pancakes_in_pan`, `plate`, `served`, etc.
        
        2. For each action in the sequence:
           - Check preconditions from the world model.
           - If **not met**, set `"state_after": "not applicable"` and do not simulate the effects.
           - If met, update the state by applying the effects.
           - Track any irreversible changes or flags as needed.
        
        3. After each action, output the current world state using:
           - **Facts**: simple state truths
           - **Objects**: e.g., contents of blender or pan
           - **Flags**: like `pancakes_cooked: true`, `blended: false`, etc.
        ---
        ### Output Format:
        
        Respond with a JSON list of steps. Each item includes:
        - `"action_index"`: integer
        - `"action"`: the action string
        - "current_state: state of the world"
        - `"state_after"`: state after applying the effect
          - a dictionary with state facts and flags (if valid), or
        - "reward": simulate the reward of rl agent environment, can it apply action on current state . 1 - indicates yes, 0-indicates no.
        -"explanation": explain the reward
        
        Example:
        
        ```json
        [
          {{
            "action_index": 0,
            "action": "START",
            "state_after": {{"process_started": true}}
          }},
          {{
            "action_index": 4,
            "action": "",
            "state_after": "not applicable"
          }}
        ]
        {action_sequence}
        """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)
        raw_response = raw_response['response']

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "action_index": 0,
                "action": [],
                "state_after": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed_states = json.loads(clean_response)

        return parsed_states

    def eval_llm_based_constrains_and(self,action_sequence, actions, goal, violated_subsequences, state_trace):
        prompt1 = f"""
        You are an expert evaluator of goal‑directed action sequences. Your job is to judge whether a given plan logically and causally achieves its goal, step by step.
        
        ### Input:
        - **GOAL**: "{goal}"
        - **ACTIONS** (in order):
          {action_sequence}
        - **NAMED STEPS**:
          A map from index to description, {actions}
        - **STATE TRACE**:
        {state_trace}
          A list of records showing, for each action:
            - action_index
            - current_state
            - action description
            - state_after
            - explanation
        - **PRECEDENCE CONSTRAINT VIOLATIONS**:
        {violated_subsequences}
          A list of pairs `(A, B)` indicating that, according to the constraint model, **B should precede A** but in the plan **A precedes B**.
        
        ### Evaluation Instructions:
        1. For each reported violation `(A, B)`:
           - Locate A and B in the state trace.
           - Ask: “Did executing A before B cause any impossible or illogical state transition?”
           - If yes, mark that violation as causing a failure; note exactly which state transition broke commonsense (e.g., “You poured batter before heating the pan”).
        2. Independently scan the state trace for any other out‑of‑order dependencies not listed.
        3. Decide **reward = 1** if **no critical violations** occur (i.e., all preconditions are met when each action runs) and the **final state** reflects the goal; otherwise **reward = 0**.
        4. In your explanation:
           - List each critical violation, its index pair, and why it breaks the goal (e.g., “Chopping strawberries after serving breaks the requirement that chopped fruit must exist when plating”).
           - Confirm whether the final state meets the goal conditions.
           - Keep it concise and refer to specific action indices and state entries.
        
        ### Output (strict JSON):
        ```json
        {{
          "reward": 0 or 1,
          "explanation": "Detailed analysis referencing action indices, violating pairs, and final state consistency."
        }}"""


        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)

        # Step 1: Check and extract the 'response' field if present
        if isinstance(raw_response, dict) and 'response' in raw_response:
            raw_response = raw_response['response']
        else:
            print("⚠️ raw_response is not a dictionary or missing 'response' key.")
            return None

        # Step 2: Remove wrapping ```json``` if present
        if isinstance(raw_response, str) and raw_response.startswith("```json"):
            raw_response = raw_response.strip("```json").strip("```").strip()

        # Step 3: Clean the raw JSON-like string
        clean_response = self.clean_json_like_string(raw_response)

        # Step 4: Try to parse it safely
        try:
            parsed_eval = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw response was:")
            print(clean_response)
            parsed_eval = None
        return parsed_eval

    def state_evaluation_no_world_model(self, action_sequence, actions, goal):
        prompt1 = f""" 
        You are a skilled simulator that tracks world state changes from sequences of high-level actions.
        Your job is to trace how the state of the world evolves when performing a given sequence of actions, step-by-step, toward achieving a goal.

        ### GOAL:
        "{goal}"

        ### Action Sequence (with indices and descriptions):
        {actions}

        ### Instructions:
        1. Start from an empty or neutral world state.
        2. For each action in the sequence:
           - Reason about what the action *physically or logically* changes.
           - Update the world state using intuitive variables, such as:
             - `blender_contents`, `batter_blended`, `pan_heated`, `pancakes_on_plate`, etc.
           - If an action doesn’t make sense given the **current state** or violates **temporal logic**, set `"state_after": "not applicable"` and explain why.
        3. Use common-sense reasoning about everyday tasks — **you do NOT need a predefined constraint model.**

        ### Temporal Logic Rule (Allen’s Interval Algebra):
        - You must ensure that actions **respect proper temporal relationships**:
          - If action X must enable or precede action Y, then **X must temporally precede Y**.
          - Use the relation **X before Y** if Y depends on the effects of X.
          - If a violation occurs , mark the state as `"not applicable"` and explain the violated relation.

        ### Output Format:
        Respond with a JSON list of steps. Each item must include:
        - `"action_index"`: the index of the action
        - `"current_state"`: dictionary of the world state before the action
        - `"action"`: the action description
        - `"state_after"`: the updated state (or `"not applicable"`)
        - `"explanation"`: what changed, or which temporal/causal rule was violated

        ### Example:
        
        ```json
        [
          {{
            "action_index": 0,
            "current_state": {{}},
            "action": "START",
            "state_after": {{"process_started": true}},
            "explanation": "Initiates the process."
          }},
          {{
            "action_index": 7,
            "current_state": {{"process_started": true}},
            "action": "Pour batter into pan",
            "state_after": "not applicable",
            "explanation": "Batter has not yet been blended."
          }}
        ]
        
        Here are the sequences to evaluate:
        [
        {action_sequence}
        ]
            """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self.query_llm_proxy(prompt)

        if isinstance(raw_response, dict):
            raw_response = raw_response.get("response", "")
        if not isinstance(raw_response, str):
            print("⚠️ raw_response is not a string.")
            return None
        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return None

        clean_response = self.clean_json_like_string(raw_response)

        # Step 4: Try to parse it safely
        try:
            parsed = json.loads(clean_response)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw response was:")
            print(clean_response)
            parsed = None

        return parsed

    def evaluate_critic(self, action_sequence, actions, goal,physics_data):
        prompt1 = f"""
        You are a critical evaluator with deep knowledge of real-world processes, procedures, and planning.
        You must determine whether a proposed sequence of actions is logically valid and physically feasible to reach the given goal.
        ---
        ### GOAL:
        "{goal}"
        ### Bag of available actions (indexed):
        {actions}
        ### Physical effects of each action (physics_data):
        Use this dictionary to reason about what each action physically changes and whether the effect is irreversible:

        {physics_data}

        ### Instructions:

        You will receive one or more sequences of action indices. For each sequence:

        1. Simulate the actions step-by-step as if performing them in the real world.
        2. At each step, retrieve the **physics** of the current action (Effect and Irreversible).
        3. Keep track of the **material state** 
        4. Before applying an action, check:
           - Is the **required physical state** present?
           - Has an **irreversible transformation** already occurred that makes the action invalid?
           - Does this step make sense at this time based on prior steps?
        5. Only give a reward of 1 if the sequence is:
           - Logically coherent
           - Physically valid according to `physics_data`
           - Free from contradictions or premature irreversible steps
        Output format:
        Respond strictly in this JSON format, using double quotes only:
        {{
          "sequence_scores": [
            {{
              "sequence": [list of integers],
              "reward": 1 or 0,
              "explanation": "..."
            }}
          ]
        }}

        Here are the sequences to evaluate:
        [
        {action_sequence}
        ]
        """
        parsed_barch = []
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)

        parsed_barch.append(parsed['sequence_scores'])

        return parsed_barch[0][0]['reward'], parsed_barch[0][0]['explanation']

    def evaluate_critic_parts(self, action_sequence, actions, goal,physics_data):
        prompt1 = f"""
        You are a critical evaluator with deep understanding of real-world processes, physical effects of actions, and procedural planning. You must determine whether a proposed sequence of actions or the part that you gave is logically valid and sufficient to reach the goal.

        GOAL:
        "{goal}"

        Bag of available actions (indexed):
        {actions}

        For each action, you also have physical consequence data:
        {physics_data}

        Each action's physics description explains the state change it causes and whether it is irreversible. Use this to carefully reason about action feasibility and order.

        You will be given one or more sequences of action indices it can be a total sequence or a few first steps. Each index corresponds to a step in the action list above.

        For each sequence:

        1. Carefully simulate the actions one by one — as if performing them in the real world.
        2. At each step, use the physics info to understand what changes and whether it's reversible.
        3. Ask yourself:
           - Does this step make physical sense at this point?
           - Is it attempting something irreversible too early?
           - Is it redundant or violates the expected physical process?
        4. Use critical thinking. Do **not** assume the sequence is valid. It must match common-sense order and physical realism.
        5. Only give a reward of 1 if the sequence **clearly and reasonably** leads to the goal in a coherent order, without contradictions or premature/illogical steps.

        Output format:
        Respond strictly in this JSON format, using double quotes only:
        {{
          "sequence_scores": [
            {{
              "sequence": [list of integers],
              "reward": 1 or 0,
              "explanation": "..."
            }}
          ]
        }}

        Here are the sequences to evaluate:
        [
        {action_sequence}
        ]
        """
        parsed_barch = []
        #
        # for i in range(l_action_sequence//batch_LLM+1):
        #     prompt = prompt1
        #     actions_text = []
        #     for seq in action_sequence[i*batch_LLM:min(l_action_sequence,(i+1)*batch_LLM)]:
        #         actions_text = [actions[str(i)] for i in seq]
        #         prompt += f"- {actions_text}\n\n"
        #         if actions_text==[]:
        #             break
        #     if actions_text == []:
        #
        #         break
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)

        parsed_barch.append(parsed['sequence_scores'])
        # parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
        # parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
        # print("bad transitions")
        # print(parsed['bad transitions'])
        # print(f"Explanation: {parsed['explanation']}\n")
        # print(f"reward: {parsed['reward']}\n")

        return parsed_barch[0][0]['reward'], parsed_barch[0][0]['explanation']

    def evaluate_seq(self, action_sequence, actions, goal):
        """
        Evaluate a final sequence of actions for chronological reasonableness.
        Return: {reward: 1 or 0, transitions: [problematic transitions] in dot format}
        """

        """
         Evaluate a final sequence of actions for chronological reasonableness.
         Return: {reward: 1 or 0, transitions: [...], explanation: "..."}
         """
        # actions_text = [str(i)+'-' + actions[str(i)] for i in action_sequence]
        # actions_text = [str(i) for i in action_sequence]
        prompt1 = (f"""
        You are an expert at recepies and in planning. Pat attention carefully for every action. Given the following goal: '{goal}', and the following bag of actions:
        {actions}

        You given a sequence of actions to reach the goal. Imagine you are simulating step be step the sequence, does all the steps lead to achieve the goal.
        Do they appear in logical ordering. 
        Simulate the sequence dis it made you closer to the goal? Return 1 if yes. Otherwise, return 0. No repetition allowed.
        Check that every step is comming at the right order. Make sure that the steps with in the sequence you get are logical. 
        
        🔧 For every sequence return the result in valid JSON format, using double quotes (") only. Do not use single quotes (') around keys or values.

        Respond exactly in the following JSON format:
        {{
          "sequence_scores": [
            {{"sequence": [list of steps], "reward": 0 or 1, "explanation":}}
            ... 
          ]
        }}

        Here are the sequences to evaluate:
        """)

        # l_action_sequence = len(action_sequence)
        # batch_LLM = min(5,l_action_sequence)
        parsed_barch = []
        #
        # for i in range(l_action_sequence//batch_LLM+1):
        #     prompt = prompt1
        #     actions_text = []
        #     for seq in action_sequence[i*batch_LLM:min(l_action_sequence,(i+1)*batch_LLM)]:
        #         actions_text = [actions[str(i)] for i in seq]
        #         prompt += f"- {actions_text}\n\n"
        #         if actions_text==[]:
        #             break
        #     if actions_text == []:
        #
        #         break
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)

        parsed_barch.append(parsed['sequence_scores'])
        # parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
        # parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
        # print("bad transitions")
        # print(parsed['bad transitions'])
        # print(f"Explanation: {parsed['explanation']}\n")
        # print(f"reward: {parsed['reward']}\n")

        return parsed_barch[0][0]['reward'],parsed_barch[0][0]['explanation']

    def actions_physics(self, actions):

        prompt = f"""
        You are a world-modeling expert.

        You will receive a list of actions taken from a real-world task (such as cooking, repairing, or assembling).  
        For each action, describe the **physical consequences** of performing it, including:

        1. The **state change** it causes in the object or material.
        2. Whether this change is **irreversible**.

        Avoid suggesting what should happen before or after — only describe what *this action* does when it is executed.
        Here are the actions: {actions}

        Return the output in a structured JSON format.

        ### Format your output like this:

        {{
          "0": {{
            "Effect": "[description of physical change]",
            "Irreversible": "Yes or No"
          }},
          "1": {{
            "Effect": "...",
            "Irreversible": "..."
          }}
        }}
        """
        parsed_barch = []
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)

        return parsed

    def evaluate_seq_parts(self, action_sequence, actions, goal):
        """
        # Evaluate a final sequence of actions for chronological reasonableness.
        # Return: {reward: 1 or 0, transitions: [problematic transitions] in dot format}
        # """
        #
        # """
        #  Evaluate a PART of sequence that should be chronological reasonableness.
        #  Return: {reward: 1 or 0, transitions: [...], explanation: "..."}
        #  """
        # # actions_text = [str(i)+'-' + actions[str(i)] for i in action_sequence]
        # # actions_text = [str(i) for i in action_sequence]
        # prompt1 = (f"""
        # You are an expert at recepies and in planning. Pat attention carefully for every action. Given the following goal: '{goal}', and the following bag of actions:
        # {actions}
        #
        # Given a part of sequence not neccasery from the begining you should be evaluated for whether the steps in the sequence are logically ordered to help to achieve the goal,
        # considering the required ordering of actions, dependencies, and logical transitions based on typical task structure. Pay attention to every transition during the sequence. Not only to start and end also to the middle of the sequence.
        # If a sequence makes logical chronological sense to achieve the goal, return 1. Otherwise, return 0. No redundancy or repetition allowed.
        #
        # 🔧 For every sequence return the result in valid JSON format, using double quotes (") only. Do not use single quotes (') around keys or values.
        #
        # Respond exactly in the following JSON format:
        # {{
        #   "sequence_scores": [
        #     {{"sequence": [list of steps], "reward": 0 or 1, "explanation":}}
        #     ...
        #   ]
        # }}
        #
        # Here are the sequences to evaluate:
        # """)

        prompt1 = f"""
        You are an expert in procedural planning and task analysis. Pay careful attention to every action transition.

        Goal: '{goal}'
        Available actions (unordered bag of steps):
        {actions}

        Task:
        You are given a **partial sequence** of actions (not necessarily starting from the beginning). Your job is to evaluate whether the order of actions in this sequence is **chronologically and logically valid** toward achieving the goal.

        The sequence must:
        - Follow a **reasonable logical and temporal structure** based on the nature of the task.
        - Avoid redundancy (e.g., repeating actions unnecessarily).
        - Respect typical **dependencies** (e.g., some actions must come before others).
        - Make sense even if it's only a partial sub-plan, not a full solution.

        Do not judge the sequence for not starting at the first step — just evaluate whether **every transition between actions** is appropriate given typical task flow.

        Output Instructions:
        Return a single JSON object using **double quotes only**. The format must be:

        Respond exactly in the following JSON format:
         {{
           "sequence_scores": [
             {{"sequence": [list of steps], "reward": 0 or 1, "explanation":}}
             ... 
           ]
         }}

        Be concise but clear in the explanation. Reward is `1` if the sequence is logically ordered and free of major planning flaws; otherwise, return `0`.

        Sequences to evaluate:
        """

        # l_action_sequence = len(action_sequence)
        # batch_LLM = min(5,l_action_sequence)
        parsed_barch = []
        #
        # for i in range(l_action_sequence//batch_LLM+1):
        #     prompt = prompt1
        #     actions_text = []
        #     for seq in action_sequence[i*batch_LLM:min(l_action_sequence,(i+1)*batch_LLM)]:
        #         actions_text = [actions[str(i)] for i in seq]
        #         prompt += f"- {actions_text}\n\n"
        #         if actions_text==[]:
        #             break
        #     if actions_text == []:
        #
        #         break
        actions_text = [actions[str(i)] for i in action_sequence]
        prompt = prompt1 + f"-{actions_text}\n\n"
        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

        clean_response = self.clean_json_like_string(raw_response)

        parsed = json.loads(clean_response)

        parsed_barch.append(parsed['sequence_scores'])
        # parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
        # parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
        # print("bad transitions")
        # print(parsed['bad transitions'])
        # print(f"Explanation: {parsed['explanation']}\n")
        # print(f"reward: {parsed['reward']}\n")

        return parsed_barch[0][0]['reward']


    def evaluate_sequence(self, action_sequence, actions, goal):
        """
        Evaluate a final sequence of actions for chronological reasonableness.
        Return: {reward: 1 or 0, transitions: [problematic transitions] in dot format}
        """
        actions_text = [actions[str(i)] for i in action_sequence]
        """
         Evaluate a final sequence of actions for chronological reasonableness.
         Return: {reward: 1 or 0, transitions: [...], explanation: "..."}
         """
        actions_text = [str(i)+'-' + actions[str(i)] for i in action_sequence]
        #actions_text = [str(i) for i in action_sequence]
        prompt = (
            f"Given the following goal: '{goal}', and the following bag of actions:\n"
            f"{actions}\n"
            "Evaluate the following sequence of actions for achieving the goal, the sequence should include all bag actions:\n"
            f"{actions_text}\n"
            "The values at the beginning of the actions and sequence is dot language. Is the sequence chronologically reasonable?  Return the answer in JSON format as follows:\n"
            "{\n"
            "  \"reward\": 1 or 0,\n"
            "  \"good transitions\": [list only in order transition in the sequence at dot language, focus only on the right next transition whether it logical. if the sequence [0,4,7,8] and 4 can be right after 0 then [0,4] is in order transition. no "" before the numbers. The format [[0,4],]],\n"
            "  \"bad transitions\":  [list only out of order transition in the sequence at dot language, focus only on the right next transition/action Not the following after whether it logical or not. if the sequence [0,4,7,8] and 4 cant be right after 0 then [0,4] is out of order transition. no "" before the numbers. The format [[0,4],]],\n"
            "  \"explanation\": \"Your reasoning here\"\n"
            "}"
        )

        raw_response = self._query_gpt(prompt)

        # Attempt to sanitize and parse the response
        try:
            # Sometimes GPT wraps the response in a code block, strip it
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

            parsed = json.loads(raw_response)
            parsed_recheck = self.recheck_bad_transitions(goal, actions, action_sequence, parsed['bad transitions'])
            parsed['bad transitions'] = parsed_recheck['confirmed_bad_transitions']
            print("bad transitions")
            print(parsed['bad transitions'])
            print(f"Explanation: {parsed['explanation']}\n")
            print(f"reward: {parsed['reward']}\n")

            return parsed

        except json.JSONDecodeError as e:
            print("❌ Failed to parse GPT response as JSON. Response was:")
            print(raw_response)
            return {
                "reward": 0,
                "transitions": [],
                "explanation": "Failed to parse GPT output"
            }

    def recheck_bad_transitions(self, goal, actions, sequence, bad_transitions):
        """
        Ask GPT to re-evaluate specific 'bad' transitions more carefully.
        """

        actions_text = [actions[str(i)] for i in sequence]
        prompt = (
            f"Given the goal: '{goal}' and the following symbolic representation of actions:\n"
            f"{actions}\n\n"
            f"The current sequence of actions (in dot IDs) was: {sequence}\n"
            f"The corresponding action names are: {actions_text}\n"
            f"The following transitions were marked as 'bad': {bad_transitions}\n\n"
            "Please recheck each bad transition one by one and evaluate if they are truly out of order.\n"
            "For each, respond with a JSON format only like this:\n"
            "{\n"
            "  \"confirmed_bad_transitions\": [[a, b], ...],\n"
            "  \"mistakenly_marked\": [[x, y], ...],\n"
            "  \"explanation\": \"Give a concise explanation per case\"\n"
            "}"
        )

        raw_response = self._query_gpt(prompt)

        try:
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip("```json").strip("```").strip()

            parsed = json.loads(raw_response)
            return parsed

        except json.JSONDecodeError as e:
            print("❌ GPT output could not be parsed. Raw response:")
            print(raw_response)
            return {
                "confirmed_bad_transitions": [],
                "mistakenly_marked": [],
                "explanation": "Failed to parse GPT output"
            }

    # gpt - 4o - mini - 2024 - 07 - 18
    #gpt - 4.1 - 2025 - 04 - 14
    def _query_gpt(self, prompt):
        """Internal helper to call the GPT API."""
        response_out = {}
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for understanding action plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            response_out["response"] = response.choices[0].message.content.strip()
            return response_out
        except Exception as e:
            return f"Error querying GPT: {str(e)}"

    def query_llm_proxy(self,prompt):
        response = generate(model='4o-mini',
                            system='You are a helpful assistant for understanding action plans.',
                            query=prompt,
                            temperature=0.0,
                            lastk=0,
                            session_id='GenericSession',
                            rag_usage=True,
                            rag_threshold=0.5,
                            rag_k=1)

        return response

    def validate_constraints(self, parsed_constraints, steps):
        """Validate that all action indices used in constraints exist in the steps."""
        valid_keys = set(map(int, steps.keys()))
        return all(
            isinstance(pair, list) and
            len(pair) == 2 and
            all(isinstance(i, int) and i in valid_keys for i in pair)
            for pair in parsed_constraints
        )

    def filter_constraints(self, constraints, steps):
        """Remove constraint pairs that include any step not in the original list of steps."""
        valid_keys = set(map(int, steps.keys()))
        return [pair for pair in constraints if
                isinstance(pair, list) and len(pair) == 2 and all(isinstance(i, int) and i in valid_keys for i in pair)]


def check_constrains_llm_eval(input_folder, save_folder, api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)
    idx = []
    edges_llm = []
    evaluation_constrains = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

        if filename.endswith(".json"):
            save_path = os.path.join(save_folder, filename)
            with open(save_path, 'r') as file:
                data_results_collection = json.load(file)


            world_model = data.get("validate_constarins_llm_proxy_4o_mini", {})
            edges = data.get("edges",{})
            steps = data.get("steps", {})
            physics = data.get("physics", {})
            goal = os.path.splitext(filename)[0]

            for target_str, data_w in world_model.items():
                target = int(target_str)
                for source_str in data_w["preconditions"]:
                    source = int(source_str)
                    edges_llm.append((source, target))
            metric_llm = Metrics(true_constrains=edges_llm)
            metric_gt = Metrics(true_constrains=edges)

            for result in data_results_collection:
                for _, info in result.items():
                    # gt_labels.append(info.get('gt_label'))
                    contrains_eval = metric_llm.check_trajectory_edges(info.get('sequence'))
                    contrains_gt = metric_gt.check_trajectory_edges(info.get('sequence'))

                    # evaluation_constrains.append(
                    #     int((metric.check_trajectory_edges(info.get('sequence')))['all_respected']))
                    state_trace = connector.state_evaluation_no_world_model(info.get('sequence'),steps, goal)
                    response = connector.eval_llm_based_constrains_and(info.get('sequence'), steps, goal, contrains_eval['violated_in_current_sequence'], state_trace)
                response['state_trace'] = state_trace
                result['constarins_eval_state_llm']=response
                # llm_rewards_direct.append(info.get('llm_reward_direct'))
                # llm_rewards_state.append(info.get('llm_reward_state_eval'))
                idx.append((info.get('entry_id')))

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data_results_collection, f, indent=4)

def main(input_folder, api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)

    output_folder = os.path.join(input_folder, "LLM")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            goal = os.path.splitext(filename)[0]

            constraints = connector.generate_constraints_dot(steps, goal)
            data["constraints_LLM"] = constraints

            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as out_file:
                json.dump(data, out_file, indent=2)

    print(f"Processed files saved to: {output_folder}")

def generate_physics(input_folder,api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            goal = os.path.splitext(filename)[0]
            physics = connector.actions_physics(steps)


            # Step 2: Add physics info to each matching action ID
            data['physics'] = physics
            #file_path = r"C:\Users\spaste01\Documents\Research\data\train_data\pp"

            # Step 3: Save the updated file
            with open(input_path, "w") as f:
                json.dump(data, f, indent=2)

            print("Physics data successfully added to the existing file.")

def generate_constrains(input_folder,api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            physics = data.get("physics", {})
            goal = os.path.splitext(filename)[0]
            constrains = connector.llm_constrains(steps,goal,physics)
            #constarins = data.get('LLM_costarins_with_physics_2',{})
            validate_constrains = connector.validate_constrains(steps, goal, physics,constrains)

            # Step 2: Add physics info to each matching action ID
            data['LLM_costarins_with_physics_llm_proxy_4o_mini'] = constrains
            data['validate_constarins_llm_proxy_4o_mini'] = validate_constrains
            #file_path = r"C:\Users\spaste01\Documents\Research\data\train_data\pp"

            # Step 3: Save the updated file
            with open(input_path, "w") as f:
                json.dump(data, f, indent=2)

            print("Physics data successfully added to the existing file.")

def generate_pddl(input_folder,save_dir,api_key=None):
    connector = GPTFeedbackConnector(api_key=api_key)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            goal = os.path.splitext(filename)[0]
            world_model = data.get('validate_constarins_llm_proxy_4o_mini',{})
            # file_path = r"C:\Users\spaste01\Documents\Research\data\train_data\pp"
        pddl_file = connector.generate_pddl(steps, goal, world_model)
        # Save to files
        save_path_domain = save_dir+'domain_'+goal+'.pddl'
        save_path_problem = save_dir+'problem_'+goal+'.pddl'
        with open(save_path_domain, "w") as f:
            f.write(pddl_file['domain'])

        with open(save_path_problem, "w") as f:
            f.write(pddl_file['problem'])

def check_contrastive(input_folder,save_dir,api_key=None):
        # fill the buffer with good examples
    permutations_path = 'C:/Users/spaste01/Documents/Research/data/backlog_data/blenderbananapancakes.json'
    with open(permutations_path, 'r') as file:
        data = json.load(file)
    permutations_seq = data['permutations_seq']

    bad_seq = [0,1,7,5,9,3,13,12,11,8,6,10,14,2,4,15]
  
    connector = GPTFeedbackConnector(api_key=api_key)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                data = json.load(file)

            steps = data.get("steps", {})
            goal = os.path.splitext(filename)[0]
            world_model = data.get('validate_constarins_llm_proxy_4o_mini',{})
            # file_path = r"C:\Users\spaste01\Documents\Research\data\train_data\pp"
        explanation = connector.trajectory_action_explanation(steps, goal, good_seq)
        pddl_file = connector.check_contrastive_prompt(steps, goal, good_seq,bad_seq)
        # Save to files
        save_path_domain = save_dir+'domain_'+goal+'.pddl'
        save_path_problem = save_dir+'problem_'+goal+'.pddl'
        with open(save_path_domain, "w") as f:
            f.write(pddl_file['domain'])

        with open(save_path_problem, "w") as f:
            f.write(pddl_file['problem'])


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Run GPT constraint generation on JSON action files.")
    # parser.add_argument("folder", help="Path to the folder with JSON files")
    # parser.add_argument("--api_key", help="OpenAI API key (optional if set in environment)")
    # args = parser.parse_args()
    folder = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/"
    save_dir = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/pddls/"
    main(input_folder=folder)
    #generate_constrains(input_folder=folder)
    # generate_pddl(input_folder=folder,save_dir = save_dir)
    #check_contrastive(input_folder=folder)
    check_constrains_llm_eval(input_folder = folder,save_folder=r"C:/Users/spaste01/PycharmProjects/Results/PPO_RL/LLM_evaluation_trajectories/",)

    y=1

