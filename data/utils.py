# PYCD/data/utils.py
import os
import json
import pandas as pd
import numpy as np
import ast


def _parse_weights(x, dim=11):
    """Parse the 'weights' column into a float list of length `dim`.
    Pads with zeros if shorter, truncates if longer.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = list(x)
    else:
        s = str(x)
        try:
            arr = json.loads(s)
        except Exception:
            try:
                arr = ast.literal_eval(s)
            except Exception:
                arr = [float(t.strip()) for t in s.strip("[]").split(",") if t.strip()]

    arr = [0.0 if v is None else float(v) for v in arr[:dim]]

    if len(arr) < dim:
        arr += [0.0] * (dim - len(arr))

    return arr


def load_a_from_csv(a_csv_path, num_exercises, a_dim=11, id_col_candidates=("question_id", "id")):
    """
    Load A-matrix from a two-column CSV: [id, weights].

    Returns:
        np.ndarray of shape [num_exercises, a_dim]
        Missing exercises are filled with zeros.
    """
    if not os.path.exists(a_csv_path):
        print(f"[A] not found: {a_csv_path}, returning zero matrix")
        return np.zeros((num_exercises, a_dim), dtype=np.float32)

    df = pd.read_csv(a_csv_path)

    id_col = None
    for c in id_col_candidates:
        if c in df.columns:
            id_col = c
            break

    if id_col is None or "weights" not in df.columns:
        raise ValueError(
            f"A CSV must contain 'id' or 'question_id' and 'weights' columns, got: {df.columns}"
        )

    A = np.zeros((num_exercises, a_dim), dtype=np.float32)

    for _, r in df.iterrows():
        qid = int(r[id_col])
        if 0 <= qid < num_exercises:
            A[qid, :] = np.array(_parse_weights(r["weights"], a_dim), dtype=np.float32)

    return A


def convert_csv_to_json(csv_path, q_matrix, out_path, a_matrix=None):
    """
    Convert CSV (with columns: user_id, question_id, correct) into JSON format.

    Each sample includes:
        - q_vector (from Q-matrix)
        - optional a_vector (from A-matrix)
    """
    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        qid = int(row['question_id'])
        uid = int(row['user_id'])
        label = float(row['correct'])

        q_vec = q_matrix[qid].tolist()

        sample = {
            'user_id': uid,
            'exer_id': qid,
            'correct': label,
            'q_vector': q_vec
        }

        if a_matrix is not None:
            sample['a_vector'] = a_matrix[qid].tolist()

        samples.append(sample)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f)

    print(f"[Convert] {csv_path} → {out_path} (with A={a_matrix is not None})")


def batch_convert_to_rcd_json(data_dir, a_csv_name="A_matrix_mapped.csv", a_dim=11):
    """
    Batch convert train_valid.csv and test.csv into JSON format.

    Required:
        - q_matrix.csv

    Optional:
        - A matrix CSV (default: 'A_matrix_mapped.csv')
    """
    qmat_path = os.path.join(data_dir, 'q_matrix.csv')
    train_valid_csv = os.path.join(data_dir, 'train_valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    train_json = os.path.join(data_dir, 'train_valid.json')
    test_json = os.path.join(data_dir, 'test.json')

    if all(os.path.exists(p) for p in [train_json, test_json]):
        print("[Skip] JSON files already exist")
        return

    # --- Build Q-matrix ---
    df = pd.read_csv(qmat_path)
    df['concept_ids'] = df['concept_ids'].apply(lambda x: ast.literal_eval(x))
    df = df.explode('concept_ids')

    max_qid = int(df['question_id'].max())
    max_kid = int(df['concept_ids'].max())

    q_matrix = np.zeros((max_qid + 1, max_kid + 1), dtype=np.float32)

    for _, row in df.iterrows():
        q_matrix[int(row['question_id']), int(row['concept_ids'])] = 1.0

    print(f"[Q] shape = {q_matrix.shape}")

    # --- Load A-matrix if available ---
    a_csv_path = os.path.join(data_dir, a_csv_name)
    a_matrix = load_a_from_csv(a_csv_path, num_exercises=q_matrix.shape[0], a_dim=a_dim)

    print(f"[A] shape = {a_matrix.shape} (file exists={os.path.exists(a_csv_path)})")

    # --- Convert to JSON ---
    convert_csv_to_json(train_valid_csv, q_matrix, train_json, a_matrix=a_matrix)
    convert_csv_to_json(test_csv, q_matrix, test_json, a_matrix=a_matrix)


def load_A_like_Q_matrix(a_csv_path: str, num_exercises: int, num_concepts: int):
    """
    Load A-matrix in the same shape as Q-matrix.

    Expected CSV format:
        question_id, weights
    """
    import os
    import ast
    import pandas as pd
    import numpy as np

    if not os.path.exists(a_csv_path):
        print(f"[A] file not found: {a_csv_path}")
        return np.zeros((num_exercises, num_concepts), dtype=np.float32)

    df = pd.read_csv(a_csv_path)

    if "question_id" not in df.columns or "weights" not in df.columns:
        raise ValueError(
            f"[A] Invalid CSV format. Expected columns ['question_id', 'weights'], got: {list(df.columns)}"
        )

    A = np.zeros((num_exercises, num_concepts), dtype=np.float32)

    def parse_weights(x):
        try:
            arr = ast.literal_eval(str(x))
            return np.array(arr, dtype=np.float32)
        except Exception:
            return np.zeros(num_concepts, dtype=np.float32)

    for _, row in df.iterrows():
        qid = int(row["question_id"])

        if not (0 <= qid < num_exercises):
            continue

        vec = parse_weights(row["weights"])

        # Pad or truncate automatically
        if len(vec) < num_concepts:
            vec = np.pad(vec, (0, num_concepts - len(vec)))
        elif len(vec) > num_concepts:
            vec = vec[:num_concepts]

        A[qid, :] = vec

    print(
        f"[A] shape={A.shape}, nnz={(A > 0).sum()}, mean={A.mean():.6f}, "
        f"from {os.path.basename(a_csv_path)}"
    )

    return A
