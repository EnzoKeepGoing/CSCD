import pandas as pd
import numpy as np
import os,json
import random
import hashlib
from collections import defaultdict
from tqdm import tqdm
from .utils import set_seed, sta_infos, write_txt, replace_text, format_list2str, split_sequences_by_time_gap, improved_smart_sequence_merge

KEYS = ["user_id", "topic", "exercise_id"]

def load_q2c(qname):
    """Load the mapping from questions to concepts
    
    Args:
        qname (str): Path to the mapping file
        
    Returns:
        dict: Dictionary of question to concept mapping
    """
    df = pd.read_csv(qname, encoding="utf-8", low_memory=False).dropna(subset=["name", "topic"])
    dq2c = dict()
    for name, topic in zip(df["name"], df["topic"]):
        if name not in dq2c:
            dq2c[name] = topic
        else:
            print(f"题目已存在于字典中: {name}: {topic}, {dq2c[name]}")
    print(f"题目到主题映射数量: {len(dq2c)}")
    return dq2c


def process_data(
    read_file: str,
    write_file: str,
    mode: int = 1,
    time_gap_weeks: int = 0,
    dq2c: dict = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    sample_size: int = 5000
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
        sample_size (int): Number of sampled users, default 5000
    """
    set_seed(seed)
    stares = []

    # -------------------
    # 1. 读取、初步统计、清洗
    # -------------------
    # 加载数据
    df = pd.read_csv(read_file)
    user_seq_lengths = df.groupby('user_id').size()
    qualified_users = user_seq_lengths[user_seq_lengths >= min_seq_len].index.tolist()
    if sample_size > 0 and sample_size < len(qualified_users):
        # 设置随机种子并从合格用户中随机抽样
        random.seed(seed)
        sampled_users = random.sample(qualified_users, sample_size)
        print(f"Sampled {len(sampled_users)} users, all users' sequence length >= {min_seq_len}")
    else:
        # 使用所有合格用户
        sampled_users = qualified_users
        print(f"Using all qualified users, total {len(sampled_users)}")
    df = df[df['user_id'].isin(sampled_users)]
    sampled_seq_lengths = df.groupby('user_id').size()
    print(f"Sampled users' average sequence length: {sampled_seq_lengths.mean():.2f}")
    print(f"Sampled users' minimum sequence length: {sampled_seq_lengths.min()}")
    print(f"Sampled users' maximum sequence length: {sampled_seq_lengths.max()}")

    # 加载题目到主题的映射，并处理题目和主题文本
    df["topic"] = df["exercise"].apply(lambda q: "NANA" if q not in dq2c else dq2c[q])
    df["exercise"] = df["exercise"].apply(replace_text)
    df["topic"] = df["topic"].apply(replace_text)
    df = df[df["topic"] != "NANA"]
    
    # 为exercise创建唯一的exercise_id（数值型）
    exercise_mapping = {ex: idx+1 for idx, ex in enumerate(df["exercise"].unique())}
    df["exercise_id"] = df["exercise"].map(exercise_mapping)
    
    # 数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"预处理后数据: 交互数: {ins}, 用户数: {us}, 问题数: {qs}, 概念数: {cs}")
    
    # 数据清洗和预处理
    df["index"] = range(df.shape[0])
    print(f"预处理记录形状: {df.shape}")
    
    df = df[["index", "user_id", "exercise", "exercise_id", "time_done", "time_taken_attempts", "correct", "count_attempts", "topic"]]
    df = df.dropna(subset=["user_id", "exercise_id", "time_done", "correct"])
    df = df[df["correct"].isin([False, True])]
    
    # 转换correct为数值型
    df["correct"] = df["correct"].map({True: 1, False: 0})
    
    # 处理时间字段
    df["time_taken_attempts"] = (df["time_taken_attempts"].fillna(-100)).astype(str)
    df.loc[:, "time_taken_attempts"] = df["time_taken_attempts"].astype(str).apply(lambda x: int(x.split("&")[0])*1000)
    df.loc[:, "time_done"] = df["time_done"].astype(int)
    
    # 确保time_done是毫秒级别的整数
    df.loc[:, "time_done"] = df["time_done"].apply(lambda x: int(str(x)[:13]))
    
    # 为主题创建一个唯一ID
    topic_mapping = {topic: idx+1 for idx, topic in enumerate(df["topic"].unique())}
    df["topic_id"] = df["topic"].map(topic_mapping)

    # 数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    
    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["user_id"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        df2 = df.sort_values(by=["user_id", "index"])
        processed_df = df2.drop_duplicates(
            subset=["user_id", "exercise_id"],
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
        corrects = user_block[user_block["correct"] == 1]["exercise_id"].unique()
        incorrects = user_block[user_block["correct"] == 0]["exercise_id"].unique()

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
                processed_df["exercise_id"] == pid
            )
            processed_df.loc[mask, "is_test"] = True

        user_test_problems[str(user_id)] = set(test_set)

    # 打印测试集分布
    tc = processed_df["is_test"].sum()
    print(f" Marked {tc} test samples ({tc/len(processed_df):.2%})")
    print(
        f"Training set accuracy: {processed_df[~processed_df['is_test']]['correct'].mean():.4f}, "
        f"Test set accuracy: {processed_df[processed_df['is_test']]['correct'].mean():.4f}"
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
            processed_df, time_gap_weeks, "user_id", "time_done"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "user_id", "time_done"
        )
        after_cnt = processed_df["user_id"].nunique()
        print(
            f"Before splitting: {before_cnt}, After splitting: {after_cnt}; "
            f"Test samples still {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "Error: Test set标记丢失！"

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
        block = block.sort_values(by=["time_done", "index"])

        seq_problems = block["exercise_id"].tolist()
        seq_skills = block["topic_id"].tolist()
        seq_ans = block["correct"].astype(int).tolist()
        seq_times = block["time_done"].tolist()
        seq_costs = block["time_taken_attempts"].tolist()
        
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
    dq2c: dict = None,
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    seed: int = 42,
    
):
    """Main entry: Call process_data based on split_mode"""
    if split_mode not in [1, 2, 3]:
        raise ValueError(f"Unsupported processing mode: {split_mode}")
    process_data(
        read_file,
        write_file,
        mode=split_mode,
        time_gap_weeks=time_gap_weeks,
        dq2c=dq2c,
        seed=seed,
        test_ratio=test_ratio,
        min_seq_len=min_seq_len,
    )


