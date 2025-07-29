from textwrap import dedent

notebook_code = dedent("""
    # Simulated Planning Comparison Notebook

    import random
    from itertools import permutations

    # -------------------------------
    # Rule-Based Planner (NeSyC-style)
    # -------------------------------

    def validate_constraints(sequence, rules):
        for (a, b), relation in rules.items():
            if relation == "must precede" and sequence.index(a) > sequence.index(b):
                return False
        return True

    def generate_plan_with_rules(actions, rules):
        for perm in permutations(actions):
            if validate_constraints(perm, rules):
                return list(perm)
        return None

    # -------------------------------
    # Few-Shot LLM-Planner Prompt Generator
    # -------------------------------

    def generate_llm_plan(task_description, few_shot_examples):
        prompt = (
            "You are an expert planner. Your goal is to provide a correct and efficient plan "
            f"to achieve the task: {task_description}.\\n\\n"
            "Here are some examples of valid plans:\\n"
        )

        for example in few_shot_examples:
            prompt += f"Task: {example['task']}\\nPlan: {example['plan']}\\n\\n"

        prompt += f"Task: {task_description}\\nPlan:"
        return prompt  # For testing; replace with actual LLM call

    # -------------------------------
    # LLM Feedback (Simulated Scoring)
    # -------------------------------

    def score_plan(plan, goal_description):
        # Simulate LLM feedback as noisy signal: reward 1 if plan ends with 'Serve', else random
        if plan and plan[-1] == "Serve":
            return 1.0
        return round(random.uniform(0.3, 0.8), 2)

    # -------------------------------
    # Simulated Task Comparison
    # -------------------------------

    goal_description = "make a smoothie"
    actions = ["Add banana", "Add milk", "Blend", "Serve"]
    rules = {
        ("Add banana", "Blend"): "must precede",
        ("Add milk", "Blend"): "must precede",
        ("Blend", "Serve"): "must precede"
    }

    # Rule-based plan
    rule_plan = generate_plan_with_rules(actions, rules)
    print("Rule-Based Plan:", rule_plan)
    print("Rule-Based Score:", score_plan(rule_plan, goal_description))

    # LLM-Planner (Few-shot)
    few_shot_examples = [
        {"task": "make a smoothie", "plan": "Add banana. Add milk. Blend. Serve."},
        {"task": "boil pasta", "plan": "Boil water. Add pasta. Cook. Drain. Serve."}
    ]
    llm_prompt = generate_llm_plan(goal_description, few_shot_examples)
    print("\\nLLM Prompt Preview:\\n", llm_prompt)

    # Simulate LLM-generated plan (for now, match example)
    llm_plan = ["Add banana", "Add milk", "Blend", "Serve"]
    print("LLM Plan:", llm_plan)
    print("LLM Plan Score:", score_plan(llm_plan, goal_description))
""")

notebook_code
