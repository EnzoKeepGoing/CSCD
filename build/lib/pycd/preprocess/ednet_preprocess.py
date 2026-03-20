import pandas as pd
import numpy as np
import random, json
import os, hashlib
from collections import defaultdict
from .utils import set_seed, sta_infos, write_txt, split_sequences_by_time_gap, format_list2str, improved_smart_sequence_merge


def load_ednet_data(read_file, dataset_name=None):
    """Load EdNet dataset
    
    Args:
        read_file (str): Input file directory
        dataset_name (str, optional): Dataset name, used for path replacement
        
    Returns:
        DataFrame: Loaded and processed data frame
    """
    # 处理路径
    if dataset_name is not None:
        write_dir = read_file.replace("/ednet/", f"/{dataset_name}")
        print(f"Data directory: {write_dir}")
    else:
        write_dir = read_file
    
    # 随机抽样用户
    random.seed(2)
    samp = [i for i in range(840473)]
    random.shuffle(samp)
    count = 0
    file_list = list()
    start_i = 0
    
    # 读取用户数据文件
    print("Loading user data...")
    for unum in samp:
        str_unum = str(unum)
        df_path = os.path.join(read_file, f"KT1/u{str_unum}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = unum
            file_list.append(df)
            count = count + 1
        if dataset_name == "ednet" and count == 5000:
            start_i = 0
            break
        elif dataset_name == "ednet5w" and count == 50000+5000:
            start_i = 5000
            break
    
    print(f"Loaded {count} users")
    all_sa = pd.concat(file_list[start_i:])
    print(f"Processed records: {len(all_sa)}")
    all_sa["index"] = range(all_sa.shape[0])
    
    # 读取问题内容文件
    ca = pd.read_csv(os.path.join(read_file, 'contents', 'questions.csv'))
    
    # 保存样本数据
    os.makedirs(write_dir, exist_ok=True)
    all_sa.to_csv(os.path.join(write_dir, 'ednet_sample.csv'), index=False)
    
    # 处理标签
    ca['tags'] = ca['tags'].apply(lambda x: x.replace(";", "_"))
    ca = ca[ca['tags'] != '-1']
    
    # 合并数据
    co = all_sa.merge(ca, sort=False, how='left')
    co = co.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    co['correct'] = (co['correct_answer'] == co['user_answer']).apply(int)
    
    # 确保时间戳是毫秒级别
    co.loc[:, 'timestamp'] = (co['timestamp']).astype(int)
    # 确保 elapsed_time 是毫秒
    co.loc[:, 'elapsed_time'] = co['elapsed_time'].astype(int)
    
    return co, write_dir

def process_data(read_file, write_file, mode=1, time_gap_weeks=0, test_ratio=0.2, min_seq_len=15, seed=42,  dataset_name=None):
    """Process EdNet data and output to TXT file, supports three different processing modes
    
    Args:
        read_file (str): Input file directory
        write_file (str): Output file path
        mode (int): Processing mode (1, 2, or 3)
        time_gap_weeks (int): Time interval threshold (weeks), greater than 0 will split sequences
        seed (int): Random seed
        alpha (float): Weight coefficient for personal accuracy in mode 3
        dataset_name (str): Dataset name, used for path replacement
        
    Returns:
        tuple: Processed write directory and file path
    """
    set_seed(seed)
    stares = []
    
    KEYS = ["user_id", "tags", "question_id"]
    
    # 处理路径
    if dataset_name is not None:
        write_file = write_file.replace("/ednet/", f"/{dataset_name}/")
        print(f"Write file: {write_file}")
    
    # 加载数据
    process_file = os.path.join(read_file.replace("/ednet/", f"/{dataset_name}" if dataset_name else "/ednet/"), 'ednet_sample_process.csv')
    if os.path.exists(process_file):
        # 如果存在预处理数据，直接读取
        co = pd.read_csv(process_file)
        write_dir = os.path.dirname(process_file)
        print(f"Loaded data from preprocessed file: {process_file}")
    else:
        # 否则加载并预处理数据
        co, write_dir = load_ednet_data(read_file, dataset_name)
        # 保存处理后的数据
        co.to_csv(os.path.join(write_dir, 'ednet_sample_process.csv'), index=False)
    
    # 数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    print(f" Original data: Interactions: {ins}, Users: {us}, Questions: {qs}, Concepts: {cs}")
    
    # 为标签创建唯一的标签ID
    tag_mapping = {tag: idx+1 for idx, tag in enumerate(co["tags"].unique())}
    co["tags_id"] = co["tags"].map(tag_mapping)
    co['index'] = range(len(co))
    df = co.copy()
     # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["user_id"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        df2 = df.sort_values(by=["user_id", "index"])
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
        corrects = user_block[user_block["correct"] == 1]["question_id"].unique()
        incorrects = user_block[user_block["correct"] == 0]["question_id"].unique()

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
            processed_df, time_gap_weeks, "user_id", "timestamp"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "user_id", "timestamp"
        )
        after_cnt = processed_df["user_id"].nunique()
        print(
            f"Before splitting: {before_cnt}, After splitting: {after_cnt}; "
            f"Test samples still {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "错误：测试集标记丢失！"

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
        block = block.sort_values(by=["timestamp", "index"])
        seq_len = len(block)
        seq_problems = block["question_id"].tolist()
        seq_skills = block["tags_id"].tolist()
        seq_ans = block["correct"].astype(int).tolist()
        seq_times = block["timestamp"].tolist()
        seq_costs = block["elapsed_time"].tolist()
        
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
        
        

def read_data_from_csv(read_file, write_file, split_mode=1, time_gap_weeks=0, test_ratio=0.2, min_seq_len=15, seed=42, dataset_name='ednet'):
    """Main function to process CSV data
    
    Args:
        read_file (str): Input file directory
        write_file (str): Output file path
        split_mode (int): Processing mode (1, 2, or 3)
        time_gap_weeks (int): Time interval threshold (weeks), used to split long sequences
        seed (int): Random seed
        dataset_name (str, optional): Dataset name, used for path replacement
    """
    if split_mode not in [1, 2, 3]:
        raise ValueError(f"Unsupported processing mode: {split_mode}")
    
    # 调用处理函数
    process_data(read_file, write_file, mode=split_mode, time_gap_weeks=time_gap_weeks, test_ratio=test_ratio, min_seq_len=min_seq_len, seed=seed, dataset_name=dataset_name)
