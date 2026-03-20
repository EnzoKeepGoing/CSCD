import pandas as pd
import numpy as np
import os
import json
import hashlib
from collections import defaultdict
from .utils import (
    set_seed,
    sta_infos,
    write_txt,
    format_list2str,
    split_sequences_by_time_gap,
    improved_smart_sequence_merge,
)

KEYS = ["stu_id", "concept_id", "que_id"]

def process_data(
    read_file: str,
    write_file: str,
    mode: int = 1,
    time_gap_weeks: int = 0,
    seed: int = 42,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    dq2c: dict = None,
):
    """Process CSV data and output to TXT file, supports three different processing modes

    Args:
        read_file (str): Path to input CSV file
        write_file (str): Path to output TXT file
        mode (int): Processing mode (1, 2, or 3)
        time_gap_weeks (int): Time gap threshold (weeks), will split sequences when greater than 0
        seed (int): Random seed
        test_ratio (float): Test set ratio, default 0.2 (20%)
        min_seq_len (int): Minimum length requirement for sequence merging, default 15
    """
    set_seed(seed)
    stares = []
    df = pd.read_csv(read_file, low_memory=False)
    # 合并知识点信息
    cs = []
    for i, row in df.iterrows():
        qid = str(row["que_id"])
        cid = dq2c[qid]
        cs.append(cid)
    df["concept_id"] = cs

    # -------------------
    # 1. 读取、初步统计、清洗
    # -------------------
    # df = pd.read_csv(read_file, encoding="utf-8", low_memory=False)
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {df.shape[0]}, "
        f"user num: {us}, question num: {qs}, concept num: {cs}, "
        f"avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["stu_id", "timestamp", "que_id", "label"])
    df = df[df['label'].isin([0,1])] #filter responses
    # df['label'] = df['label'].astype(int)


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, "
        f"question num: {qs}, concept num: {cs}, "
        f"avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["stu_id"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        df2 = df.sort_values(by=["stu_id", "index"])
        processed_df = df2.drop_duplicates(
            subset=["stu_id", "que_id"],
            keep="first",
            ignore_index=True
        )
    elif mode in [2, 3]:
        raise NotImplementedError(f"Mode {mode} is not implemented, please use mode 1")
    else:
        raise ValueError(f"Unsupported processing mode: {mode}")

    # 过滤掉序列长度<min_seq_len的学生
    student_seq_lens = processed_df.groupby('stu_id').size()
    valid_students = student_seq_lens[student_seq_lens >= min_seq_len].index
    processed_df = processed_df[processed_df['stu_id'].isin(valid_students)]
    print(f"Before filtering: {len(student_seq_lens)}, After filtering: {len(valid_students)}, Removed {len(student_seq_lens) - len(valid_students)} students")
    # -------------------
    # 3. 分层抽取测试集
    # -------------------
    processed_df["is_test"] = False
    user_test_problems = {}

    for user_id in processed_df["stu_id"].unique():
        user_block = processed_df[processed_df["stu_id"] == user_id]
        corrects = user_block[user_block["label"] == 1]["que_id"].unique()
        incorrects = user_block[user_block["label"] == 0]["que_id"].unique()

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
            mask = (processed_df["stu_id"] == user_id) & (
                processed_df["que_id"] == pid
            )
            processed_df.loc[mask, "is_test"] = True

        user_test_problems[str(user_id)] = set(test_set)

    # 打印测试集分布
    tc = processed_df["is_test"].sum()
    print(f"Marked {tc} test samples ({tc/len(processed_df):.2%})")
    print(
        f"Training set accuracy: {processed_df[~processed_df['is_test']]['label'].mean():.4f}, "
        f"Test set accuracy: {processed_df[processed_df['is_test']]['label'].mean():.4f}"
    )

    # 保存测试集详情
    # test_info_path = os.path.join(os.path.dirname(write_file), "test_problems.json")
    # os.makedirs(os.path.dirname(write_file), exist_ok=True)
    # with open(test_info_path, "w") as f:
    #     out = {
    #         k: [int(x) if hasattr(x, "item") else x for x in sorted(list(v))]
    #         for k, v in user_test_problems.items()
    #     }
    #     json.dump(out, f, indent=2)

    # -------------------
    # 4. 时间间隔切分 & 合并（可选）
    # -------------------
    if time_gap_weeks > 0:
        before_cnt = processed_df["stu_id"].nunique()
        processed_df, _ = split_sequences_by_time_gap(
            processed_df, time_gap_weeks, "stu_id", "timestamp"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "stu_id", "timestamp"
        )
        after_cnt = processed_df["stu_id"].nunique()
        print(
            f"Before splitting: {before_cnt}, After splitting: {after_cnt}; "
            f"Test samples still {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "错误：测试集标记丢失！"

    # -------------------
    # 5. 按「原始出现顺序 + 后缀」生成最终 UID 列表
    # -------------------
    processed_df["stu_id"] = processed_df["stu_id"].astype(str)
    all_ids = processed_df["stu_id"].unique().tolist()

    def sort_key(uid: str):
        parts = uid.split("_")
        base = parts[0]
        suffix = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        return (orig_order.index(base), suffix)

    processed_students = sorted(all_ids, key=sort_key)

    # -------------------
    # 6. 组装输出 lines
    # -------------------
    user_inters = []
    for uid in processed_students:
        block = processed_df[processed_df["stu_id"] == uid].copy()
        # 按时间戳 + 原始 index 排序
        block = block.sort_values(by=["timestamp", "index"])

        seq_problems = block["que_id"].tolist()
        seq_skills = block["concept_id"].tolist()
        seq_ans = block["label"].astype(int).tolist()
        seq_times = block["timestamp"].tolist()
        seq_costs = ["NA"]
        seq_is_test = block["is_test"].astype(int).tolist()
        length = len(seq_problems)

        user_inters.append(
            [
                [uid, str(length)],
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
    dq2c: dict = None,
    seed: int = 42
):
    """Main entry: Call process_data based on split_mode"""
    if split_mode not in [1, 2, 3]:
        raise ValueError(f"Unsupported processing mode: {split_mode}")
    process_data(
        read_file,
        write_file,
        mode=split_mode,
        time_gap_weeks=time_gap_weeks,
        seed=seed,
        test_ratio=test_ratio,
        min_seq_len=min_seq_len,
        dq2c=dq2c,
    )
