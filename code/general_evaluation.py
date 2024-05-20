from sklearn.metrics import normalized_mutual_info_score
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
#python general_evaluation.py --measure nmi --ground ./data/ground_truth/1-6.cmty --detected ./data/louvain/1-6.cmty

# Reading the Ground-Truth Community Data
def load_communities(file_path):
    node_to_community = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node, community = int(parts[0]), int(parts[1])
                node_to_community[node] = community
    # Convert to list of lists for compatibility with NMI calculation
    community_to_nodes = {}
    for node, community in node_to_community.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values()), node_to_community


# Calculating NMI Score
def calculate_nmi(true_communities, detected_communities):
    true_labels = {}
    for i, community in enumerate(true_communities):
        for node in community:
            true_labels[node] = i
    detected_labels = {}
    for i, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node] = i

    nodes = sorted(set(true_labels) | set(detected_labels))
    true_labels_vector = [true_labels[node] for node in nodes]
    detected_labels_vector = [detected_labels.get(node, -1) for node in nodes]

    return normalized_mutual_info_score(true_labels_vector, detected_labels_vector)


# Calculating F1 Score with Hungarian Algorithm for optimal matching
def calculate_f1(true_node_to_community, detected_node_to_community):
    nodes = sorted(set(true_node_to_community) | set(detected_node_to_community))
    true_labels_vector = [true_node_to_community.get(node, -1) for node in nodes]
    detected_labels_vector = [detected_node_to_community.get(node, -1) for node in nodes]

    unique_true_labels = np.unique(true_labels_vector)
    unique_detected_labels = np.unique(detected_labels_vector)
    confusion_matrix = np.zeros((len(unique_true_labels), len(unique_detected_labels)))

    for i, true_label in enumerate(unique_true_labels):
        for j, detected_label in enumerate(unique_detected_labels):
            confusion_matrix[i, j] = np.sum(
                (true_labels_vector == true_label) & (detected_labels_vector == detected_label))

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    best_matching = confusion_matrix[row_ind, col_ind]

    tp = np.sum(best_matching)
    fp = np.sum(confusion_matrix) - tp
    fn = np.sum(confusion_matrix) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1


def eval(ground_truth_file_path, detected_communities_file_path, measure):
    true_communities, true_node_to_community = load_communities(ground_truth_file_path)
    detected_communities, detected_node_to_community = load_communities(detected_communities_file_path)

    if measure == 'nmi':
        return calculate_nmi(true_communities, detected_communities)
    elif measure == 'f1':
        return calculate_f1(true_node_to_community, detected_node_to_community)
    else:
        raise ValueError("Unsupported measure. Please choose either 'nmi', 'f1', or 'nvi'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate community detection in a network.')
    parser.add_argument('--ground', '-g', type=str, help='The path of the ground truth community file.', required=True)
    parser.add_argument('--detected', '-d', type=str, help='The path of the detected communities file.', required=True)
    parser.add_argument('--measure', '-m', type=str, help='The measure to evaluate (nmi, f1, or nvi).', required=True)
    args = parser.parse_args()

    ground_truth_file_path = args.ground
    detected_communities_file_path = args.detected
    measure = args.measure

    result = eval(ground_truth_file_path, detected_communities_file_path, measure)
    print(f'{measure.upper()} score: {result}')
