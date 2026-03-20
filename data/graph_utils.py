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
from data.dataset import load_q_matrix,load_question_mapping

def ensure_rcd_graph_files(dataset_name: str, data_dir: str = "data"):
    """确保RCD模型所需的特定图文件存在"""
    print("Checking RCD required graph files...")
    
    graph_dir = os.path.join(data_dir, dataset_name, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    
    # RCD需要的特定文件
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
    
    # 按依赖顺序构建
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
    构建知识点之间的依赖图（KC-KC graph），输出为 Graph_KC-KC.txt。

    Args:
        dataset_name (str): 如 "assist2009"
        data_dir (str): 数据主目录，默认 data
    """
    dataset_path = os.path.join(data_dir, dataset_name)
    data_path = os.path.join(dataset_path, "data.txt")
    mapping_path = os.path.join(dataset_path, "id_mapping.json")
    output_dir = os.path.join(dataset_path, "graph")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "Graph_KC-KC.txt")

    with open(data_path, 'r') as f:
        lines = f.readlines()

    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)
        kc2id = id_mapping["concepts"]  # dict: raw_kc_str -> mapped_id (int)

    students_data = []
    i = 0
    while i < len(lines):
        student_id_line = lines[i].strip()
        question_ids = lines[i + 1].strip().split(',')
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
    对 Graph_KC-KC.txt 中的边进行处理，拆分为单向边（KC-KC_Directed.txt）和共现边（KC-KC_Undirected.txt）
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

def construct_kc_ques_graph(dataset_name: str, data_dir: str = "data"):
    dataset_path = os.path.join(data_dir, dataset_name)
    data_path = os.path.join(dataset_path, "data.txt")
    mapping_path = os.path.join(dataset_path, "id_mapping.json")
    output_dir = os.path.join(dataset_path, "graph")
    os.makedirs(output_dir, exist_ok=True)

    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)
        ques2id = id_mapping["questions"]
        kc2id = id_mapping["concepts"]

    exer_n = len(ques2id)
    q_from_k = set()
    k_from_q = set()

    with open(data_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        question_ids = lines[i + 1].strip().split(',')
        kc_sequences = lines[i + 2].strip().split(',')
        for qid_raw, kc_str in zip(question_ids, kc_sequences):
            if qid_raw not in ques2id:
                continue
            qid = ques2id[qid_raw]
            for kc_raw in kc_str.strip().split('_'):
                if kc_raw not in kc2id:
                    continue
                kid = kc2id[kc_raw] + exer_n
                q_from_k.add((kid, qid))
                k_from_q.add((qid, kid))
        i += 7

    with open(os.path.join(output_dir, "Graph_K_from_Q.txt"), 'w') as f:
        for i, j in k_from_q:
            f.write(f"{i}\t{j}\n")

    with open(os.path.join(output_dir, "Graph_Q_from_K.txt"), 'w') as f:
        for i, j in q_from_k:
            f.write(f"{i}\t{j}\n")

    print(f"[construct_kc_ques_graph] Finished. {len(k_from_q)} Q→K and {len(q_from_k)} K→Q edges")

def construct_stu_ques_graph(dataset_name: str, data_dir: str = "data"):
    dataset_path = os.path.join(data_dir, dataset_name)
    data_path = os.path.join(dataset_path, "data.txt")
    mapping_path = os.path.join(dataset_path, "id_mapping.json")
    output_dir = os.path.join(dataset_path, "graph")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(mapping_path, 'r') as f:
        id_mapping = json.load(f)
        stu2id = id_mapping["uid"]
        ques2id = id_mapping["questions"]

    exer_n = len(ques2id)
    u_from_q = set()
    q_from_u = set()

    with open(data_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        uid = lines[i].strip().split(',')[0]
        if uid not in stu2id:
            i += 7
            continue
        sid = stu2id[uid] + exer_n
        ques_seq = lines[i + 1].strip().split(',')
        for q in ques_seq:
            if q in ques2id:
                qid = ques2id[q]
                u_from_q.add((qid, sid))
                q_from_u.add((sid, qid))
        i += 7

    with open(os.path.join(output_dir, "Graph_Stu_from_Q.txt"), 'w') as f:
        for i, j in u_from_q:
            f.write(f"{i}\t{j}\n")

    with open(os.path.join(output_dir, "Graph_Q_from_Stu.txt"), 'w') as f:
        for i, j in q_from_u:
            f.write(f"{i}\t{j}\n")

    print(f"[construct_stu_ques_graph] Finished. {len(u_from_q)} Q→U and {len(q_from_u)} U→Q edges")

def construct_graph(dataset_name: str, graph_type: str, num_nodes: int, data_dir: str = "data"):
    """
    构建 DGL 图对象，根据图类型加载图文件。

    Args:
        dataset_name (str): 数据集名称
        graph_type (str): ['direct', 'undirect', 'k_from_e', 'e_from_k', 'u_from_e', 'e_from_u']
        num_nodes (int): 图节点总数（由调用者提供）
        data_dir (str): 数据主目录

    Returns:
        dgl.DGLGraph: 构建好的图对象
    """
    file_map = {
        'direct': 'KC-KC_Directed.txt',
        'undirect': 'KC-KC_Undirected.txt',
        'k_from_e': 'Graph_K_from_Q.txt',
        'e_from_k': 'Graph_Q_from_K.txt',
        'u_from_e': 'Graph_Stu_from_Q.txt',
        'e_from_u': 'Graph_Q_from_Stu.txt'
    }
    assert graph_type in file_map, f"Unsupported graph type: {graph_type}"

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

    # 对于 undirected 图类型，反向再添加一次边
    if graph_type == 'undirect':
        g.add_edges(dst, src)

    print(f"[build_graph] Graph '{graph_type}' with {g.num_nodes()} nodes and {g.num_edges()} edges loaded.")
    return g

def get_node_count_for_graph(graph_type: str, dataset_name: str, data_dir: str = "data") -> int:
    """
    根据图类型返回所需的节点数：
    - 'direct' / 'undirect' => knowledge_n
    - 'k_from_e' / 'e_from_k' => knowledge_n + exer_n
    - 'u_from_e' / 'e_from_u' => student_n + exer_n
    """
    mapping_path = os.path.join(data_dir, dataset_name, "id_mapping.json")
    with open(mapping_path, 'r') as f:
        id_map = json.load(f)

    knowledge_n = len(id_map.get("concepts", {}))
    exer_n = len(id_map.get("questions", {}))
    student_n = len(id_map.get("uid", {}))

    if graph_type in ['direct', 'undirect']:
        return knowledge_n
    elif graph_type in ['k_from_e', 'e_from_k']:
        return knowledge_n + exer_n
    elif graph_type in ['u_from_e', 'e_from_u']:
        return student_n + exer_n
    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'")

def construct_local_map(dataset_name: str,data_dir: str = "data"):
    return {
        'directed_g': construct_graph(dataset_name, 'direct', get_node_count_for_graph('direct', dataset_name,data_dir), data_dir),
        'undirected_g': construct_graph(dataset_name, 'undirect', get_node_count_for_graph('undirect', dataset_name,data_dir), data_dir),
        'k_from_e': construct_graph(dataset_name, 'k_from_e', get_node_count_for_graph('k_from_e', dataset_name,data_dir), data_dir),
        'e_from_k': construct_graph(dataset_name, 'e_from_k', get_node_count_for_graph('e_from_k', dataset_name,data_dir), data_dir),
        'u_from_e': construct_graph(dataset_name, 'u_from_e', get_node_count_for_graph('u_from_e', dataset_name,data_dir), data_dir),
        'e_from_u': construct_graph(dataset_name, 'e_from_u', get_node_count_for_graph('e_from_u', dataset_name,data_dir), data_dir),
    }


def disengcd_get_file(dataset_name: str, data_dir: str = "data"):
    dataset_path = os.path.join(data_dir, dataset_name)
    output_dir = os.path.join(dataset_path, "graph")
    #data_len = 22032
  
    rows1 = []
    cols1 = []
    with open(os.path.join(output_dir,'Graph_K_from_Q.txt'), 'r') as f1:
      for line in f1.readlines():
          row, col = line.strip().split('\t')  
          rows1.append(int(row))
          cols1.append(int(col))

    rows1 = np.array(rows1, dtype=np.int64)
    cols1 = np.array(cols1, dtype=np.int64)
    data1 = np.ones(len(rows1))
    matrix1 = sp.sparse.coo_matrix((data1,(rows1,cols1)))    
    matrix2 = sp.sparse.coo_matrix((data1,(cols1,rows1)))    


    rows2 = []
    cols2 = []
    with open(os.path.join(output_dir,'Graph_Q_from_Stu.txt'), 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows2.append(int(line[0]))
            cols2.append(int(line[1]))
    data2 = np.ones(len(rows2))
    matrix3 = sp.sparse.coo_matrix((data2,(rows2,cols2)))  
    matrix4 = sp.sparse.coo_matrix((data2,(cols2,rows2)))   

    rows3 = []
    cols3 = []
    with open(os.path.join(output_dir,'KC-KC_Directed.txt'), 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows3.append(int(line[0]))
            cols3.append(int(line[1]))
    data3 = np.ones(len(rows3))
    matrix5 = sp.sparse.coo_matrix((data3, (rows3, cols3)))
    rows4 = []
    cols4 = []
    with open(os.path.join(output_dir,'KC-KC_Undirected.txt'), 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows4.append(int(line[0]))
            cols4.append(int(line[1]))
    data4 = np.ones(len(rows4))
    matrix6 = sp.sparse.coo_matrix((data4, (rows4, cols4)))

    sparse_matrices = [matrix1,matrix2,matrix3,matrix4]
    with open(os.path.join(data_dir,dataset_name,"graph","edges.pkl"), 'wb') as file:
        pickle.dump(sparse_matrices, file)