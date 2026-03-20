import pandas as pd
import numpy as np
import os,json
import random
import hashlib
from collections import defaultdict
from tqdm import tqdm
from .utils import set_seed, sta_infos, write_txt, replace_text, format_list2str, change2timestamp, split_sequences_by_time_gap, improved_smart_sequence_merge
KEYS = ["user_id", "kc", "question_id"]

def process_knowledge_list(kc):
    """
    Handling knowledge list strings, unifying formats and replacing commas, underscores
    """
    return kc.replace(',', '@@@@').replace('，', '@@@@')     # for grade=4-5 and 7
    
def process_data(
    read_file: str,
    write_file: str,
    mode: int = 1,
    time_gap_weeks: int = 0,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    seed: int = 42,
):
    """Process CSV data and output to TXT file, supports three different processing modes

    Args:
        read_file (str): Input CSV file path
        write_file (str): Output TXT file path
        mode (int): Processing mode (1, 2, or 3)
        time_gap_weeks (int): Time interval threshold (weeks), greater than 0 will split sequences
        seed (int): Random seed
        test_ratio (float): Test set ratio, default 0.2 (20%)
        min_seq_len (int): Minimum length requirement for sequence merging, default 15
    """
    set_seed(seed)
    stares = []

    # -------------------
    # 1. 读取、初步统计、清洗
    # -------------------
    # 读取数据
    df = pd.read_csv(read_file, encoding='utf-8', low_memory=False)
    
    # 处理用户ID中的下划线
    df.loc[:, 'user_id'] = df['user_id'].apply(replace_text)  # 将下划线替换为'####'
    
    # 原始数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["user_id"].astype(str).unique().tolist()
    
    # 数据清洗
    df['tmp_index'] = range(len(df))
    _df = df.copy()
    
    # 处理知识点列表
    _df.loc[:, 'kc'] = _df['kc'].apply(process_knowledge_list)
    unique_kc = _df['kc'].unique().tolist()
    print(len(unique_kc), "unique knowledge points found")

    _df = _df[_df['kc'] != ""]
    _df = _df.dropna(subset=['user_id', 'question_id', 'kc', 'is_correct', 'created_at'])
    
    _df.loc[:, 'question_id'] = _df['question_id'].astype(str).apply(replace_text)
    # 验证数据格式
    assert all(_df['is_correct'].isin([0, 1])), "答案格式不正确"
    
    # 转换时间戳
    _df.loc[:, 'created_at'] = _df.loc[:, 'created_at'].apply(lambda t: change2timestamp(t, False))
    _df['is_correct'] = _df['is_correct'].astype(float)  # 确保可以计算平均值
    
    # 清洗后的数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 将处理后的数据赋回给df，确保后续处理使用正确的数据
    df = _df.copy()
    
    
    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["user_id"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        df2 = df.sort_values(by=["user_id", "tmp_index"])
        processed_df = df2.drop_duplicates(
            subset=["user_id", "question_id"],
            keep="first",
            ignore_index=True
        )
    elif mode in [2, 3]:
        raise NotImplementedError(f"Mode {mode} is not implemented, please use mode 1")
    else:
        raise ValueError(f"Unsupported processing mode: {mode}")

    # 过滤掉序列长度<min_seq_len的学生
    student_seq_lens = processed_df.groupby('user_id').size()
    valid_students = student_seq_lens[student_seq_lens >= min_seq_len].index
    processed_df = processed_df[processed_df['user_id'].isin(valid_students)]
    print(f"Before filtering: {len(student_seq_lens)}, After filtering: {len(valid_students)}, Removed {len(student_seq_lens) - len(valid_students)} students")
    
    # -------------------
    # 3. 分层抽取测试集
    # -------------------
    processed_df["is_test"] = False
    user_test_problems = {}

    for user_id in processed_df["user_id"].unique():
        user_block = processed_df[processed_df["user_id"] == user_id]
        corrects = user_block[user_block["is_correct"] == 1]["question_id"].unique()
        incorrects = user_block[user_block["is_correct"] == 0]["question_id"].unique()

        h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2**32
        rng = np.random.RandomState(h ^ seed)

        n_corr = (
            max(1, round(len(corrects) * test_ratio)) if len(corrects) > 0 else 0
        )
        n_incorr = (
            max(1, round(len(incorrects) * test_ratio)) if len(incorrects) > 0 else 0
        )
        sample_corr = (
            rng.choice(corrects, size=n_corr, replace=False) if n_corr > 0 else []
        )
        sample_incorr = (
            rng.choice(incorrects, size=n_incorr, replace=False) if n_incorr > 0 else []
        )
        test_set = np.concatenate([sample_corr, sample_incorr]) if (
            len(sample_corr) > 0 and len(sample_incorr) > 0
        ) else (sample_corr if len(sample_corr) > 0 else sample_incorr)

        for pid in test_set:
            mask = (processed_df["user_id"] == user_id) & (
                processed_df["question_id"] == pid
            )
            processed_df.loc[mask, "is_test"] = True

        user_test_problems[str(user_id)] = set(test_set)

    # 打印测试集分布
    tc = processed_df["is_test"].sum()
    print(f" Marked {tc} test samples ({tc/len(processed_df):.2%})")
    print(
        f"Training set accuracy: {processed_df[~processed_df['is_test']]['is_correct'].mean():.4f}, "
        f"Test set accuracy: {processed_df[processed_df['is_test']]['is_correct'].mean():.4f}"
    )

    # 保存测试集详情
    # test_info_path = os.path.join(os.path.dirname(write_file), "test_problems.json")
    # os.makedirs(os.path.dirname(write_file), exist_ok=True)
    # with open(test_info_path, "w") as f:
    #     out = {
    #         k: [str(x) if isinstance(x, str) else (int(x) if hasattr(x, "item") else x) for x in sorted(list(v))]
    #         for k, v in user_test_problems.items()
    #     }
    #     json.dump(out, f, indent=2)

    # -------------------
    # 4. 时间间隔切分 & 合并（可选）
    # -------------------
    if time_gap_weeks > 0:
        before_cnt = processed_df["user_id"].nunique()
        processed_df, _ = split_sequences_by_time_gap(
            processed_df, time_gap_weeks, "user_id", "created_at"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "user_id", "created_at"
        )
        after_cnt = processed_df["user_id"].nunique()
        print(
            f"Before splitting: {before_cnt}, After splitting: {after_cnt}; "
            f"Test samples still {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "Error: Test set mark lost!"

    # -------------------
    # 5. 按「原始出现顺序 + 后缀」生成最终 UID 列表
    # -------------------
    processed_df["user_id"] = processed_df["user_id"].astype(str)
    all_ids = processed_df["user_id"].unique().tolist()

    def sort_key(uid: str):
        parts = uid.split("_")
        base = parts[0]
        suffix = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        try:
            base_idx = orig_order.index(base)
        except ValueError:
            # 如果在原始顺序中找不到（可能是切分后的新ID），放到最后
            base_idx = len(orig_order)
        return (base_idx, suffix)

    processed_students = sorted(all_ids, key=sort_key)

    # -------------------
    # 6. 组装输出 lines
    # -------------------
    user_inters = []
    for uid in processed_students:
        block = processed_df[processed_df["user_id"] == uid].copy()
        # 按时间戳 + 原始 index 排序
        block = block.sort_values(by=["created_at", "tmp_index"])
        seq_len = len(block)
        seq_problems = block["question_id"].tolist()
        seq_skills = block["kc"].tolist()
        seq_ans = block["is_correct"].astype(int).tolist()
        seq_times = block["created_at"].tolist()
        seq_costs = ['NA'] * seq_len
        
        seq_is_test = block["is_test"].astype(int).tolist()
        

        user_inters.append(
            [
                [uid, str(seq_len)],
                format_list2str(seq_problems),
                format_list2str(seq_skills),
                format_list2str(seq_ans),
                format_list2str(seq_times),
                format_list2str(seq_costs),
                format_list2str(seq_is_test),
            ]
        )

    # -------------------
    # 7. 写文件
    # -------------------
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    if os.path.exists(write_file):
        os.remove(write_file)
    write_txt(write_file, user_inters)
    print(f"Data processing completed, results saved to {write_file}")


def read_data_from_csv(
    read_file: str,
    write_file: str,
    split_mode: int,
    time_gap_weeks: int = 0,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    seed: int = 42,
    
):
    """主入口：根据 split_mode 调用 process_data"""
    if split_mode not in [1, 2, 3]:
        raise ValueError(f"不支持的处理模式: {split_mode}")
    process_data(
        read_file,
        write_file,
        mode=split_mode,
        time_gap_weeks=time_gap_weeks,
        test_ratio=test_ratio,
        min_seq_len=min_seq_len,
        seed=seed,
    )

