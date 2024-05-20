import os
import argparse
import networkx as nx
from collections import defaultdict

# python detection.py --network ./dataset/TC1-6 --mode normal
# python detection.py --network ./dataset/TC1-1 --mode purity
def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

def load_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            node_id, community = line.strip().split()
            ground_truth[int(node_id)] = int(community)
    return ground_truth

def calculate_purity(node, ground_truth):
    category_count = defaultdict(int)
    for vertex in node.nodes:
        category_count[ground_truth[vertex]] += 1
    total_vertices = sum(category_count.values())
    purity = max(category_count.values()) / total_vertices if total_vertices > 0 else 0
    return category_count, purity
# tree construction 부분.
class Node:
    def __init__(self, name, type, parent=None, nodes=None):
        self.name = name
        self.type = type
        self.parent = parent
        self.children = []
        self.nodes = nodes if nodes else set()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

def construct_seed_tree(G, initial_k=3):
    root = Node("root", type="Intermediate node", nodes=set(G.nodes()))
    truss = nx.k_truss(G, initial_k) #3-truss
    if truss.number_of_nodes() == 0:
        return print("there is no 3-truss")
    root.nodes = set(G.nodes()).difference(truss.nodes()) #3-truss에 속하지 않는 것만 가짐.
    components = list(nx.connected_components(truss))
    for i, comp in enumerate(components):
        r_tree(G.subgraph(comp), initial_k, root)
    return root

def r_tree(G, k, parent):
    node = Node(f"{k}-truss", type='Intermediate node', parent=parent, nodes=set(G.nodes()))
    parent.children.append(node)

    truss = nx.k_truss(G, k+1)
    if truss.number_of_nodes() == 0:
        node.type = 'Seed node'
        return

    node.nodes = set(G.nodes()).difference(truss.nodes())
    # print(f"node name: {node.name}, node type: {node.type}, node nodes: {node.nodes}")

    components = list(nx.connected_components(truss))
    for i, comp in enumerate(components):
        r_tree(G.subgraph(comp), k+1, node)

    if len(node.children) == 1 and node.children[0].type == 'Seed node':
        node.nodes = node.nodes.union(node.children[0].nodes)
        node.children = []
        node.type = 'Seed node'

def calculate_modularity_increase(G, community, node):
    m = G.number_of_edges()  # Number of edges in the graph
    degree_node = G.degree(node)  # Degree of the node to be added
    edges_inside = sum(1 for neighbor in G.neighbors(node) if neighbor in community)
    sum_degrees_community = sum(G.degree(n) for n in community)
    expected_edges = (degree_node * sum_degrees_community) / (2 * m)
    modularity_increase = (edges_inside / m) - (expected_edges / (2 * m))
    return modularity_increase

def PostOrderIter(node):
    result = []
    def recurse(n):
        for child in n.children:
            recurse(child)
        result.append(n)
    recurse(node)
    # print(f"result: {result}")
    return result

def assign_nodes_in_intermediate_nodes(G, root):
    for node in PostOrderIter(root):
        if node.type == 'Intermediate node':
            seeds = [leaf for leaf in PostOrderIter(node) if leaf.type == 'Seed node']
            while(node.nodes):
                best_gain = 0
                best_seed = None
                v = node.nodes.pop()
                for seed in seeds:
                    increase = calculate_modularity_increase(G, seed.nodes, v)
                    if increase > best_gain:
                        best_gain = increase
                        best_seed = seed
                if best_seed != None:
                    best_seed.nodes.add(v)
                else:
                    node.nodes.add(v)

def write_community_file(tree_root, file_path):
    community_number = 1
    with open(file_path, 'w') as file:
        for node in PostOrderIter(tree_root):
            if node.type == 'Seed node':
                for member_node in node.nodes:
                    file.write(f"{member_node}\t{community_number}\n")
                community_number += 1

# Main script

def main(args):
    network_path = os.path.join(args.network, 'network.txt')
    dataset = os.path.basename(args.network)
    G = load_network(network_path)
    root = construct_seed_tree(G)
    assign_nodes_in_intermediate_nodes(G, root)

    if args.mode == 'normal':
        output_dir = f"./output/{dataset}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset}.cmty")
        write_community_file(root, output_file)
    # elif args.mode == 'purity':
    #     ground_truth_path = f"./groundtruth/{dataset}/{dataset}.cmty"
    #     ground_truth = load_ground_truth(ground_truth_path)
    #     for pre, fill, node in RenderTree(root):
    #         if not node.children:  # Only calculate purity for leaf nodes
    #             category_count, purity = calculate_purity(node, ground_truth)
    #             print(f"{pre}{node.name}: Purity = {purity:.2f}, Category Counts = {dict(category_count)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community detection using k-truss decomposition.")
    parser.add_argument("--network", type=str, required=True,
                        help="Path to the dataset directory containing the network file.")
    parser.add_argument("--mode", type=str, default='normal', choices=['normal', 'purity'], help="Mode of operation.")

    args = parser.parse_args()
    main(args)
