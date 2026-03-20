# PYCD/data/dataset.py
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_question_mapping(mapping_path: str) -> dict:
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(orig): int(mapped) for orig, mapped in data['questions'].items()}


def load_a_matrix(
    a_matrix_path: str,
    question2idx: dict = None,
    num_exercises: int = None,
    dim: int = 11,
) -> np.ndarray:

    import ast
    df = pd.read_csv(a_matrix_path)

    id_col = "id" if "id" in df.columns else ("question_id" if "question_id" in df.columns else None)
    if id_col is None or "weights" not in df.columns:
        raise ValueError(
            f"A-matrix CSV must contain 'id' or 'question_id' and 'weights' columns, got: {df.columns}"
        )

    if num_exercises is None:
        if question2idx is not None:
            num_exercises = len(question2idx)
        else:
            num_exercises = int(df[id_col].max()) + 1

    A = np.zeros((num_exercises, dim), dtype=np.float32)

    def parse_weights(x):
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

    for _, r in df.iterrows():
        raw_id = r[id_col]
        wts = parse_weights(r["weights"])

        if question2idx is not None:
            key = str(raw_id)
            if key not in question2idx:
                continue
            qidx = int(question2idx[key])
        else:
            qidx = int(raw_id)

        if 0 <= qidx < num_exercises:
            A[qidx, :dim] = np.array(wts, dtype=np.float32)

    return A


def load_q_matrix(
    q_matrix_path: str,
    question2idx: dict = None,
    num_exercises: int = None,
    num_concepts: int = None,
    auto_compress: bool = False
) -> np.ndarray:

    df = pd.read_csv(q_matrix_path)

    if 'question_id' not in df.columns or 'concept_ids' not in df.columns:
        raise ValueError(f"CSV must contain 'question_id' and 'concept_ids', got: {df.columns}")

    if num_exercises is None:
        if question2idx is not None:
            num_exercises = len(question2idx)
        else:
            num_exercises = df['question_id'].max() + 1

    all_concepts = set()
    concept_usage = {}
    rows = []

    for _, row in df.iterrows():
        item_id = row['question_id']
        try:
            concept_str = row['concept_ids']

            if isinstance(concept_str, str):
                try:
                    concept_ids = json.loads(concept_str.replace("'", '"'))
                except:
                    try:
                        concept_ids = eval(concept_str)
                    except:
                        concept_ids = [int(c.strip()) for c in concept_str.strip('[]').split(',') if c.strip()]
            else:
                concept_ids = concept_str

            if not isinstance(concept_ids, list):
                concept_ids = [concept_ids]

            all_concepts.update(concept_ids)

            for cid in concept_ids:
                concept_usage[cid] = concept_usage.get(cid, 0) + 1

            if question2idx is not None:
                if str(item_id) in question2idx:
                    mapped_id = question2idx[str(item_id)]
                else:
                    print(f"ID {item_id} not found in mapping")
                    continue
            else:
                mapped_id = item_id

            rows.append((mapped_id, concept_ids))

        except Exception as e:
            print(f"Error processing row {row}: {e}")

    if num_concepts is None:
        if all_concepts:
            num_concepts = max(all_concepts) + 1
        else:
            num_concepts = 1
            print("Warning: No concept IDs found")

    # Build multi-hot Q-matrix
    q_matrix = np.zeros((num_exercises, num_concepts), dtype=np.float32)

    for item_id, concept_ids in rows:
        for concept_id in concept_ids:
            if concept_id < num_concepts:
                q_matrix[item_id, concept_id] = 1.0

    # Check usage
    used_concepts = np.sum(q_matrix, axis=0) > 0
    actual_used_count = np.sum(used_concepts)

    unused_concepts = np.where(np.sum(q_matrix, axis=0) == 0)[0]

    # Compress matrix if necessary
    if auto_compress and actual_used_count < num_concepts * 0.8:
        nonzero_cols = np.where(used_concepts)[0]
        compressed_q_matrix = q_matrix[:, nonzero_cols]
        return compressed_q_matrix

    return q_matrix


class RCDDataset(Dataset):
    """
    Dataset for RCD tasks.
    Returns (student_idx, exercise_idx, q_vector, label).
    """

    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        user_id = data['user_id']
        exer_id = data['exer_id']
        q_vector = torch.tensor(data['q_vector'], dtype=torch.float)
        label = data['correct']
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(exer_id, dtype=torch.long),
            q_vector,
            torch.tensor(label, dtype=torch.float)
        )


class CDMDataset(Dataset):
    """
    Dataset for cognitive diagnosis tasks.

    Returns:
        (student_idx, exercise_idx, q_vector, a_vector, label)

    Supports cross-validation.
    """

    def __init__(
        self,
        csv_path: str,
        question2idx: dict,
        q_matrix: np.ndarray,
        user2idx: dict = None,
        normalize_label: bool = True,
        fold_mode: str = None,
        fold: int = None,
        is_test: bool = False,
        a_matrix: np.ndarray = None,
    ):
        df = pd.read_csv(csv_path)

        # Cross-validation filtering
        if not is_test and fold_mode and fold is not None and 'fold' in df.columns:
            if fold_mode == 'valid':
                df = df[df['fold'] == fold].copy()
            elif fold_mode == 'train':
                df = df[df['fold'] != fold].copy()

            if len(df) == 0:
                raise ValueError(f"Dataset is empty after filtering! fold_mode={fold_mode}, fold={fold}")

        df['q_idx'] = df['question_id'].astype(int)

        if user2idx is not None:
            df['u_idx'] = df['user_id'].astype(int)
        else:
            df['u_idx'] = df['user_id'].astype(int)

        df = df.dropna(subset=['q_idx', 'u_idx', 'correct'])
        df['q_idx'] = df['q_idx'].astype(int)
        df['u_idx'] = df['u_idx'].astype(int)

        self.user_idxs = df['u_idx'].values
        self.q_idxs = df['q_idx'].values

        labels = df['correct'].astype(np.float32).values
        if normalize_label:
            max_label = labels.max()
            if max_label > 1.0:
                labels = labels / max_label
        self.labels = labels

        self.q_matrix = torch.from_numpy(q_matrix)
        self.a_matrix = torch.from_numpy(a_matrix) if a_matrix is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.user_idxs[idx]
        q = self.q_idxs[idx]
        label = self.labels[idx]
        q_vector = self.q_matrix[q]

        if self.a_matrix is not None:
            a_vector = self.a_matrix[q]
            return (
                torch.tensor(u, dtype=torch.long),
                torch.tensor(q, dtype=torch.long),
                q_vector,
                a_vector,
                torch.tensor(label, dtype=torch.float),
            )
        else:
            return (
                torch.tensor(u, dtype=torch.long),
                torch.tensor(q, dtype=torch.long),
                q_vector,
                torch.tensor(label, dtype=torch.float),
            )


def get_datasets(
    mapping_path: str,
    q_matrix_path: str,
    item_count: int,
    concept_count: int,
    train_valid_csv: str,
    test_csv: str,
    fold: int = 0,
    a_matrix_path: str = None,
    a_dim: int = 11
):
    question2idx = load_question_mapping(mapping_path)

    q_matrix = load_q_matrix(
        q_matrix_path,
        None,
        num_exercises=len(question2idx),
        num_concepts=None
    )

    # Load A-matrix (aligned with Q-matrix)
    a_matrix = None
    if a_matrix_path is not None:
        a_matrix = load_a_matrix(
            a_matrix_path,
            question2idx=question2idx,
            num_exercises=len(question2idx),
            dim=a_dim
        )

    train_ds = CDMDataset(
        train_valid_csv,
        question2idx,
        q_matrix,
        fold_mode='train',
        fold=fold,
        a_matrix=a_matrix
    )

    valid_ds = CDMDataset(
        train_valid_csv,
        question2idx,
        q_matrix,
        fold_mode='valid',
        fold=fold,
        a_matrix=a_matrix
    )

    test_ds = CDMDataset(
        test_csv,
        question2idx,
        q_matrix,
        is_test=True,
        a_matrix=a_matrix
    )

    return train_ds, valid_ds, test_ds
