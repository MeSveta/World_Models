
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner,AnytimePlanner

save_dir = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/pddls/"
domain_pddl = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/pddls/domain_blenderbananapancakes.pddl"
problem_pddl = r"C:/Users/spaste01/Documents/Research/data/train_data_llm_proxy/pddls/problem_blenderbananapancakes.pddl"

# Path to the actual executable if you built it
fd_path = r"C:\Users\spaste01\PycharmProjects\LLM_check\downward\fast-downward.py"

# Read domain & problem
reader = PDDLReader()
problem = reader.parse_problem(domain_pddl, problem_pddl)

# Create a planner that works with this problem
with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    if result.plan is not None:
        for action in result.plan.actions:
            print(action)
    else:
        print("No plan found.")
y=1