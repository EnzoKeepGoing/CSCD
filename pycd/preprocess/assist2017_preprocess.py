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

KEYS = ["studentId", "skill", "problemId"]


def process_data(
    read_file: str,
    write_file: str,
    mode: int = 1,
    time_gap_weeks: int = 0,
    seed: int = 42,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
):
    """处理CSV数据并输出到TXT文件，支持三种不同的处理模式

    Args:
        read_file (str): 输入CSV文件路径
        write_file (str): 输出TXT文件路径
        mode (int): 处理模式 (1, 2, 或 3)
        time_gap_weeks (int): 时间间隔阈值(周)，大于0时会切分序列
        seed (int): 随机种子
        test_ratio (float): 测试集比例，默认0.2（20%）
        min_seq_len (int): 序列合并时的最小长度要求，默认15
    """
    set_seed(seed)
    stares = []

    # -------------------
    # 1. 读取、初步统计、清洗
    # -------------------
    df = pd.read_csv(read_file, encoding="utf-8", low_memory=False)
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {df.shape[0]}, "
        f"user num: {us}, question num: {qs}, concept num: {cs}, "
        f"avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    df["index"] = range(len(df))
    df = df.dropna(subset=["studentId", "problemId", "correct", "skill", "startTime"])
    df = df[df["correct"].isin([0, 1])]

    # 转换时间单位 & 确保 correct 是浮点
    df["timeTaken"] = df["timeTaken"].apply(lambda x: round(x * 1000))
    df["startTime"] = df["startTime"].apply(lambda t: int(t) * 1000)
    df["correct"] = df["correct"].astype(float)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, "
        f"question num: {qs}, concept num: {cs}, "
        f"avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["studentId"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        sorted_data = df.sort_values(
            by=["studentId", "problemId", "startTime", "index"]
        )
        processed_records = []
        seen = set()
        for _, row in sorted_data.iterrows():
            key = (row["studentId"], row["problemId"])
            if key not in seen:
                processed_records.append(row.to_dict())
                seen.add(key)
        processed_df = pd.DataFrame(processed_records)
    elif mode in [2, 3]:
        raise NotImplementedError(f"模式{mode}尚未实现，请使用模式1")
    else:
        raise ValueError(f"不支持的处理模式: {mode}")

    # 过滤掉序列长度<min_seq_len的学生
    student_seq_lens = processed_df.groupby('studentId').size()
    valid_students = student_seq_lens[student_seq_lens >= min_seq_len].index
    processed_df = processed_df[processed_df['studentId'].isin(valid_students)]
    print(f"过滤前学生数: {len(student_seq_lens)}, 过滤后学生数: {len(valid_students)}, 移除了 {len(student_seq_lens) - len(valid_students)} 名学生")
    # -------------------
    # 3. 分层抽取测试集
    # -------------------
    processed_df["is_test"] = False
    user_test_problems = {}

    for user_id in processed_df["studentId"].unique():
        user_block = processed_df[processed_df["studentId"] == user_id]
        corrects = user_block[user_block["correct"] == 1]["problemId"].unique()
        incorrects = user_block[user_block["correct"] == 0]["problemId"].unique()

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
            mask = (processed_df["studentId"] == user_id) & (
                processed_df["problemId"] == pid
            )
            processed_df.loc[mask, "is_test"] = True

        user_test_problems[str(user_id)] = set(test_set)

    # 打印测试集分布
    tc = processed_df["is_test"].sum()
    print(f"已标记 {tc} 个测试样本（{tc/len(processed_df):.2%}）")
    print(
        f"训练集正确率: {processed_df[~processed_df['is_test']]['correct'].mean():.4f}, "
        f"测试集正确率: {processed_df[processed_df['is_test']]['correct'].mean():.4f}"
    )

    # 保存测试集详情
    test_info_path = os.path.join(os.path.dirname(write_file), "test_problems.json")
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    with open(test_info_path, "w") as f:
        out = {
            k: [int(x) if hasattr(x, "item") else x for x in sorted(list(v))]
            for k, v in user_test_problems.items()
        }
        json.dump(out, f, indent=2)

    # -------------------
    # 4. 时间间隔切分 & 合并（可选）
    # -------------------
    if time_gap_weeks > 0:
        before_cnt = processed_df["studentId"].nunique()
        processed_df, _ = split_sequences_by_time_gap(
            processed_df, time_gap_weeks, "studentId", "startTime"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "studentId", "startTime"
        )
        after_cnt = processed_df["studentId"].nunique()
        print(
            f"切分前用户数 {before_cnt}，切分后用户数 {after_cnt}；"
            f"测试样本仍为 {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "错误：测试集标记丢失！"

    # -------------------
    # 5. 按「原始出现顺序 + 后缀」生成最终 UID 列表
    # -------------------
    processed_df["studentId"] = processed_df["studentId"].astype(str)
    all_ids = processed_df["studentId"].unique().tolist()

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
        block = processed_df[processed_df["studentId"] == uid].copy()
        # 按时间戳 + 原始 index 排序
        block = block.sort_values(by=["startTime", "index"])

        seq_problems = block["problemId"].tolist()
        seq_skills = block["skill"].tolist()
        seq_ans = block["correct"].astype(int).tolist()
        seq_times = block["startTime"].tolist()
        seq_costs = block["timeTaken"].tolist()
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
    print(f"数据处理完成，结果已保存至 {write_file}")


def read_data_from_csv(
    read_file: str,
    write_file: str,
    split_mode: int,
    time_gap_weeks: int = 0,
    test_ratio: float = 0.2,
    seed: int = 42,
    min_seq_len: int = 15,
):
    """主入口：根据 split_mode 调用 process_data"""
    if split_mode not in [1, 2, 3]:
        raise ValueError(f"不支持的处理模式: {split_mode}")
    process_data(
        read_file,
        write_file,
        mode=split_mode,
        time_gap_weeks=time_gap_weeks,
        seed=seed,
        test_ratio=test_ratio,
        min_seq_len=min_seq_len,
    )
