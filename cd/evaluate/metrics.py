# PYCD/pycd/evaluate/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import torch
import joblib
from tqdm import tqdm
import pandas as pd


def multi_metrics(model, trues, preds, extra_params=None):
    """
    Return multiple evaluation metrics (excluding DOA).
    """
    results = {
        "AUC": auc(model, trues, preds, extra_params),
        "ACC": accuracy(model, trues, preds, extra_params),
        "RMSE": rmse(model, trues, preds, extra_params)
    }
    return results


def accuracy(model, trues, preds, extra_params, threshold=0.5):
    """
    Classification accuracy.

    Args:
        trues: list or array of true labels (0 or 1)
        preds: list or array of predicted probabilities
        threshold: probability threshold for binary decision

    Returns:
        float: accuracy score
    """
    preds_bin = [p >= threshold for p in preds]
    print('ACC:', accuracy_score(trues, preds_bin))
    return accuracy_score(trues, preds_bin)


def auc(model, trues, preds, extra_params):
    """
    Area Under the ROC Curve (AUC).

    Args:
        trues: list or array of true labels (0 or 1)
        preds: list or array of predicted probabilities

    Returns:
        float: AUC score
    """
    print('AUC:', roc_auc_score(trues, preds))
    return roc_auc_score(trues, preds)


def rmse(model, trues, preds, extra_params):
    """
    Root Mean Squared Error (RMSE).

    Args:
        trues: list or array of true values
        preds: list or array of predicted values

    Returns:
        float: RMSE
    """
    print('RMSE:', mean_squared_error(trues, preds, squared=False))
    return mean_squared_error(trues, preds, squared=False)


def doa(model, trues, preds, extra_params, mode="sample_approx"):
    """
    Degree of Agreement (DOA) evaluation.

    Measures consistency between predicted mastery levels and observed responses.
    """

    eval_logs_source = extra_params["eval_logs"]

    # --- Load logs ---
    if isinstance(eval_logs_source, str):
        try:
            logs = pd.read_csv(eval_logs_source)
        except FileNotFoundError:
            print(f"Error: Log file not found at {eval_logs_source}")
            return float('nan')
    elif isinstance(eval_logs_source, pd.DataFrame):
        logs = eval_logs_source.copy()
    else:
        raise ValueError("eval_logs must be a CSV file path or a pandas DataFrame.")

    required_cols = ['user_id', 'question_id', 'correct']
    for col in required_cols:
        if col not in logs.columns:
            print(f"Error: Missing required column '{col}' in log data.")
            return float('nan')
        logs[col] = pd.to_numeric(logs[col], errors='coerce')

    logs.dropna(subset=required_cols, inplace=True)

    mas_level = model.get_all_knowledge_emb()

    # Check DOA support
    if mas_level is None:
        print(f"Warning: {model.__class__.__name__} does not support DOA calculation")
        return 0.0

    if torch.is_tensor(mas_level):
        mas_level = mas_level.detach().cpu().numpy()

    q_matrix = extra_params["q_matrix"]

    n_students = mas_level.shape[0]
    n_exercises = q_matrix.shape[0]

    r_matrix = np.full((n_students, n_exercises), -1, dtype=int)

    if mode == "sample_approx":
        method_func = calculate_doa_refined_k
    elif mode == "approx":
        method_func = calculate_doa_approx_k
    elif mode == "original":
        method_func = calculate_doa_original_k

    if not logs.empty:
        s_ids = (logs['user_id'].values - 1)
        e_ids = (logs['question_id'].values - 1)
        r_values = logs['correct'].values.astype(np.float32)

        valid_s_mask = (s_ids >= 0) & (s_ids < n_students)
        valid_e_mask = (e_ids >= 0) & (e_ids < n_exercises)
        valid_mask = valid_s_mask & valid_e_mask

        s_ids_valid = s_ids[valid_mask].astype(int)
        e_ids_valid = e_ids[valid_mask].astype(int)

        if s_ids_valid.size > 0:
            r_matrix[s_ids_valid, e_ids_valid] = r_values[valid_mask]

    know_n = q_matrix.shape[1]

    # Parallel computation
    doa_k_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(method_func)(mas_level, q_matrix, r_matrix, k)
        for k in tqdm(range(know_n), desc="Calculating DOA")
    )

    return np.mean([val for val in doa_k_list if not np.isnan(val)])


# ==========================================
# 1. Original DOA
# ==========================================
def calculate_doa_original_k(mas_level, q_matrix, r_matrix, k):
    """
    Original DOA computation (exact but slow).
    """
    n_students, _ = mas_level.shape

    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    student_pairs = np.argwhere(delta_matrix)

    question_hask = np.where(q_matrix[:, k] != 0)[0]

    if len(student_pairs) == 0 or len(question_hask) == 0:
        return 0.0

    ratios = []

    for a, b in student_pairs:
        num_ab = 0
        den_ab = 0

        for j in question_hask:
            if r_matrix[a, j] != -1 and r_matrix[b, j] != -1:
                if r_matrix[a, j] != r_matrix[b, j]:
                    den_ab += 1
                    if r_matrix[a, j] > r_matrix[b, j]:
                        num_ab += 1

        if den_ab > 0:
            ratios.append(num_ab / den_ab)

    return np.mean(ratios) if ratios else 0.0


# ==========================================
# 2. Approximate DOA
# ==========================================
def calculate_doa_approx_k(mas_level, q_matrix, r_matrix, k):
    """
    Fast approximate DOA computation.
    """
    numerator = 0
    denominator = 0

    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)

    question_hask = np.where(q_matrix[:, k] != 0)[0]

    for j in question_hask:
        attempted_mask = r_matrix[:, j] != -1
        mask_2d = np.logical_and(attempted_mask.reshape(-1, 1),
                                 attempted_mask.reshape(1, -1))

        r_j = r_matrix[:, j]
        delta_r_matrix = r_j.reshape(-1, 1) > r_j.reshape(1, -1)
        I_matrix = r_j.reshape(-1, 1) != r_j.reshape(1, -1)

        delta_matrix = delta_matrix.cpu().numpy() if isinstance(delta_matrix, torch.Tensor) else delta_matrix

        numerator_j = np.logical_and(delta_matrix, np.logical_and(mask_2d, delta_r_matrix))
        denominator_j = np.logical_and(delta_matrix, np.logical_and(mask_2d, I_matrix))

        numerator += np.sum(numerator_j)
        denominator += np.sum(denominator_j)

    return (numerator / denominator) if denominator != 0 else 0.0


# ==========================================
# 3. Refined DOA (Sampling-based)
# ==========================================
def calculate_doa_refined_k(mas_level, q_matrix, r_matrix, k, sample_size=1000):
    """
    Refined DOA using sampling-based correction.
    """
    doa_approx = calculate_doa_approx_k(mas_level, q_matrix, r_matrix, k)

    if hasattr(mas_level, 'cpu'):
        mas_level_np = mas_level.cpu().numpy()
    else:
        mas_level_np = mas_level

    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    all_pairs = np.argwhere(delta_matrix)

    if len(all_pairs) < sample_size:
        sample_pairs = all_pairs
    else:
        sample_indices = np.random.choice(len(all_pairs), size=sample_size, replace=False)
        sample_pairs = all_pairs[sample_indices]

    question_hask = np.where(q_matrix[:, k] != 0)[0]

    if len(sample_pairs) == 0 or len(question_hask) == 0:
        return doa_approx

    p_samples = []
    D_samples = []

    for a, b in sample_pairs:
        num_ab = 0
        den_ab = 0

        for j in question_hask:
            if r_matrix[a, j] != -1 and r_matrix[b, j] != -1:
                if r_matrix[a, j] != r_matrix[b, j]:
                    den_ab += 1
                    if r_matrix[a, j] > r_matrix[b, j]:
                        num_ab += 1

        if den_ab > 0:
            p_samples.append(num_ab / den_ab)
            D_samples.append(den_ab)

    if not p_samples:
        return doa_approx

    p_samples = np.array(p_samples)
    D_samples = np.array(D_samples)

    E_p = np.mean(p_samples)
    E_D = np.mean(D_samples)

    if E_D == 0:
        return doa_approx

    cov_p_D = np.mean(p_samples * D_samples) - E_p * E_D
    correction_term = cov_p_D / E_D

    doa_refined = doa_approx - correction_term

    return doa_refined
