# PYCD/data/graph_utils.py
import os
import json
import dgl
import ast
import torch
import numpy as np
from typing import List
import scipy as sp
import pickle
from data.dataset import load_q_matrix, load_question_mapping


def ensure_rcd_graph_files(dataset_name: str, data_dir: str = "data"):
    """Ensure that all required graph files for the RCD model exist."""
    print("Checking RCD required graph files...")

    graph_dir = os.path.join(data_dir, dataset_name, "graph")
    os.makedirs(graph_dir, exist_ok=True)

    required_files = {
        "KC-KC_Directed.txt": "Knowledge-Knowledge Directed Graph",
        "KC-KC_Undirected.txt": "Knowledge-Knowledge Undirected Graph",
        "Graph_K_from_Q.txt": "Exercise-Knowledge Directed Graph",
        "Graph_Q_from_K.txt": "Knowledge-Exercise Directed Graph",
        "Graph_Stu_from_Q.txt": "Exercise-Student Directed Graph",
        "Graph_Q_from_Stu.txt": "Student-Exercise Directed Graph"
    }

    missing_files = []
    for file_name, desc in required_files.items():
        file_path = os.path.join(graph_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append((file_name, desc))

    if not missing_files:
        print("All RCD graph files already exist")
        return

    print(f"Missing {len(missing_files)} graph files:")
    for file_name, desc in missing_files:
        print(f"  - {file_name} ({desc})")

    # Build graphs in dependency order
    if any("KC-KC" in f[0] for f in missing_files):
        print("→ Constructing knowledge dependency graph...")
        construct_kc_kc_graph(dataset_name, data_dir)
        process_edge(dataset_name, data_dir)

    if any("K_from_Q" in f[0] or "Q_from_K" in f[0] for f in missing_files):
        print("→ Constructing knowledge-exercise graph...")
        construct_kc_ques_graph(dataset_name, data_dir)

    if any("Stu_from_Q" in f[0] or "Q_from_Stu" in f[0] for f in missing_files):
        print("→ Constructing student-exercise graph...")
        construct_stu_ques_graph(dataset_name, data_dir)

    print("All RCD graph files constructed successfully")


def construct_kc_kc_graph(dataset_name: str, data_dir: str = "data"):
    """
    Construct the knowledge dependency graph (KC-KC graph).

    Output:
        Graph_KC-KC.txt
    """

    dataset_path = os.path.join(data_dir, dataset_name)
    data_path = os.path.join(dataset_path, "data.txt")
    mapping_path = os.path.join(dataset_path, "id_mapping.json")
    output_dir = os.path.join(dataset_path, "graph")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Graph_KC-KC.txt")

    with open(data_path, 'r') as f:
        lines = f.readlines()

    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)
        kc2id = id_mapping["concepts"]

    students_data = []
    i = 0
    while i < len(lines):
        kc_sequences = lines[i + 2].strip().split(',')
        correct_sequences = list(map(int, lines[i + 3].strip().split(',')))
        students_data.append((kc_sequences, correct_sequences))
        i += 7

    def extract_kcs(kc_str: str) -> List[int]:
        raw_ids = kc_str.strip().split('_')
        return [kc2id[r] for r in raw_ids if r in kc2id]

    knowledge_n = max(kc2id.values()) + 1
    knowledgeCorrect = np.zeros([knowledge_n, knowledge_n])
    edge_dic_deno = {}

    for kc_seq, correct_seq in students_data:
        if len(kc_seq) < 2:
            continue
        for t in range(len(kc_seq) - 1):
            if correct_seq[t] * correct_seq[t + 1] == 1:
                pre_kcs = extract_kcs(kc_seq[t])
                next_kcs = extract_kcs(kc_seq[t + 1])
                for ki in pre_kcs:
                    for kj in next_kcs:
                        if ki != kj:
                            knowledgeCorrect[ki][kj] += 1.0
                            edge_dic_deno[ki] = edge_dic_deno.get(ki, 0) + 1

    knowledgeDirected = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if i != j and knowledgeCorrect[i][j] > 0:
                knowledgeDirected[i][j] = knowledgeCorrect[i][j] / edge_dic_deno[i]

    o = np.zeros([knowledge_n, knowledge_n])
    values = []
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                values.append(knowledgeDirected[i][j])

    if not values:
        print("No valid KC-KC edges found.")
        return

    min_c = min(values)
    max_c = max(values)

    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                o[i][j] = (knowledgeDirected[i][j] - min_c) / (max_c - min_c)

    avg = np.mean(o[o > 0])
    threshold = avg ** 4

    edge_list = []
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if o[i][j] >= threshold:
                edge_list.append((i, j))

    with open(output_file, 'w') as f:
        for i, j in edge_list:
            f.write(f"{i}\t{j}\n")

    print(f"[construct_kc_kc_graph] Finished. {len(edge_list)} edges written to {output_file}")


def process_edge(dataset_name: str, data_dir: str = "data"):
    """
    Split KC-KC edges into:
    - Directed edges
    - Undirected (co-occurrence) edges
    """

    graph_dir = os.path.join(data_dir, dataset_name, "graph")
    input_file = os.path.join(graph_dir, "Graph_KC-KC.txt")
    directed_file = os.path.join(graph_dir, "KC-KC_Directed.txt")
    undirected_file = os.path.join(graph_dir, "KC-KC_Undirected.txt")

    edge = []
    with open(input_file, 'r') as f:
        for line in f:
            i, j = line.strip().split('\t')
            edge.append((i, j))

    visit = set()
    directed_edges = []
    undirected_edges = []

    for e in edge:
        if e not in visit:
            if (e[1], e[0]) in edge:
                undirected_edges.append(e)
                visit.add(e)
                visit.add((e[1], e[0]))
            else:
                directed_edges.append(e)
                visit.add(e)

    with open(directed_file, 'w') as f:
        for i, j in directed_edges:
            f.write(f"{i}\t{j}\n")

    with open(undirected_file, 'w') as f:
        for i, j in undirected_edges:
            f.write(f"{i}\t{j}\n")

    print(f"[process_edge] Directed edges: {len(directed_edges)}, Undirected edges: {len(undirected_edges)}")


def construct_graph(dataset_name: str, graph_type: str, num_nodes: int, data_dir: str = "data"):
    """
    Build a DGL graph object based on graph type.
    """

    file_map = {
        'direct': 'KC-KC_Directed.txt',
        'undirect': 'KC-KC_Undirected.txt',
        'k_from_e': 'Graph_K_from_Q.txt',
        'e_from_k': 'Graph_Q_from_K.txt',
        'u_from_e': 'Graph_Stu_from_Q.txt',
        'e_from_u': 'Graph_Q_from_Stu.txt'
    }

    graph_path = os.path.join(data_dir, dataset_name, "graph", file_map[graph_type])

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    edge_list = []
    with open(graph_path, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split('\t'))
            edge_list.append((src, dst))

    if len(edge_list) == 0:
        print(f"[construct_graph] No edges loaded from {graph_path}")
        return g

    src, dst = zip(*edge_list)
    g.add_edges(src, dst)

    if graph_type == 'undirect':
        g.add_edges(dst, src)

    print(f"[build_graph] Graph '{graph_type}' with {g.num_nodes()} nodes and {g.num_edges()} edges loaded.")
    return g


def construct_local_map(dataset_name: str, data_dir: str = "data"):
    return {
        'directed_g': construct_graph(dataset_name, 'direct', get_node_count_for_graph('direct', dataset_name, data_dir), data_dir),
        'undirected_g': construct_graph(dataset_name, 'undirect', get_node_count_for_graph('undirect', dataset_name, data_dir), data_dir),
        'k_from_e': construct_graph(dataset_name, 'k_from_e', get_node_count_for_graph('k_from_e', dataset_name, data_dir), data_dir),
        'e_from_k': construct_graph(dataset_name, 'e_from_k', get_node_count_for_graph('e_from_k', dataset_name, data_dir), data_dir),
        'u_from_e': construct_graph(dataset_name, 'u_from_e', get_node_count_for_graph('u_from_e', dataset_name, data_dir), data_dir),
        'e_from_u': construct_graph(dataset_name, 'e_from_u', get_node_count_for_graph('e_from_u', dataset_name, data_dir), data_dir),
    }


def disengcd_get_file(dataset_name: str, data_dir: str = "data"):
    """
    Build sparse adjacency matrices required for DisenGCD
    and save them as edges.pkl
    """

    dataset_path = os.path.join(data_dir, dataset_name)
    output_dir = os.path.join(dataset_path, "graph")

    rows1, cols1 = [], []
    with open(os.path.join(output_dir, 'Graph_K_from_Q.txt'), 'r') as f1:
        for line in f1:
            row, col = line.strip().split('\t')
            rows1.append(int(row))
            cols1.append(int(col))

    data1 = np.ones(len(rows1))
    matrix1 = sp.sparse.coo_matrix((data1, (rows1, cols1)))
    matrix2 = sp.sparse.coo_matrix((data1, (cols1, rows1)))

    rows2, cols2 = [], []
    with open(os.path.join(output_dir, 'Graph_Q_from_Stu.txt'), 'r') as f:
        for line in f:
            row, col = line.strip().split('\t')
            rows2.append(int(row))
            cols2.append(int(col))

    data2 = np.ones(len(rows2))
    matrix3 = sp.sparse.coo_matrix((data2, (rows2, cols2)))
    matrix4 = sp.sparse.coo_matrix((data2, (cols2, rows2)))

    sparse_matrices = [matrix1, matrix2, matrix3, matrix4]

    with open(os.path.join(output_dir, "edges.pkl"), 'wb') as f:
        pickle.dump(sparse_matrices, f)
