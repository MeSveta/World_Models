import json
import os
from collections import defaultdict, deque
import networkx as nx

def extract_constrains(sequences):
    # Step 1: Build a directed graph of all consistent pairwise precedences
    precedence_counts = defaultdict(int)
    total_sequences = len(sequences)

    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                precedence_counts[(a, b)] += 1

    # Add only the edges that are consistent across all sequences
    G = nx.DiGraph()
    for (a, b), count in precedence_counts.items():
        if count == total_sequences:
            G.add_edge(a, b)

    # Step 2: Perform transitive reduction to get minimal constraints
    minimal_graph = nx.transitive_reduction(G)

    # Extract edges
    minimal_constraints = list(minimal_graph.edges())

    return minimal_constraints
def generate_and_save_valid_sequence(filepath):
    # Load JSON data
    with open(filepath, 'r') as file:
        data = json.load(file)

    edges = data["edges"]
    steps = data["steps"]

    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # Include all nodes from steps even if they are disconnected
    for node in steps.keys():
        nodes.add(int(node))

    queue = deque()
    for node in sorted(nodes):  # Sorting ensures deterministic output
        if in_degree[node] == 0:
            queue.append(node)

    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(steps):
        raise ValueError(f"Graph in {filepath} contains a cycle or disconnected components")

    # Save the result back into the file
    data["valid_sequence"] = result

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)

    print(f"✅ Valid sequence added to: {filepath}")


def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                generate_and_save_valid_sequence(filepath)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")


# Example usage
if __name__ == "__main__":

    folder =  "C:/Users/Sveta/PycharmProjects/data/Cook/LLM"
    main(folder)
