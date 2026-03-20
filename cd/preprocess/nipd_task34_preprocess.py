import pandas as pd
import numpy as np
import os,json
import random
import hashlib
from collections import defaultdict
from tqdm import tqdm
from .utils import set_seed, sta_infos, write_txt, replace_text, format_list2str, change2timestamp, split_sequences_by_time_gap, improved_smart_sequence_merge

KEYS = ["UserId", "SubjectId_level3_str", "QuestionId"]

def load_nips_data(primary_data_path, meta_data_dir, task_name):
    """加载NIPS数据集
    
    数据来源: https://competitions.codalab.org/competitions/25449 
    文档: https://arxiv.org/abs/2007.12061

    Args:
        primary_data_path (str): 主数据路径
        meta_data_dir (str): 元数据目录
        task_name (str): 任务名称，如task_1_2或task_3_4

    Returns:
        DataFrame: 合并后的数据框
    """
    print("开始加载数据")
    answer_metadata_path = os.path.join(meta_data_dir, f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir, f"question_metadata_{task_name}.csv")
    subject_metadata_path = os.path.join(meta_data_dir, f"subject_metadata.csv")
    
    df_primary = pd.read_csv(primary_data_path)
    print(f"主数据长度: {len(df_primary)}")
    
    # 添加时间戳
    df_answer = pd.read_csv(answer_metadata_path)
    df_answer['answer_timestamp'] = df_answer['DateAnswered'].apply(change2timestamp)
    # 确保时间戳是毫秒级别的
    if not ((df_answer['answer_timestamp'] > 1000000000000).any()):
        df_answer['answer_timestamp'] = (df_answer['answer_timestamp'] * 1000).astype(int)
    
    df_question = pd.read_csv(question_metadata_path)
    df_subject = pd.read_csv(subject_metadata_path)
    
    # 只保留level 3
    keep_subject_ids = set(df_subject[df_subject['Level'] == 3]['SubjectId'])
    df_question['SubjectId_level3'] = df_question['SubjectId'].apply(lambda x: set(eval(x)) & keep_subject_ids)
    
    # 合并数据
    df_merge = df_primary.merge(df_answer[['AnswerId', 'answer_timestamp']], how='left')  # 合并答题时间
    df_merge = df_merge.merge(df_question[["QuestionId", "SubjectId_level3"]], how='left')  # 合并问题主题
    df_merge['SubjectId_level3_str'] = df_merge['SubjectId_level3'].apply(lambda x: "_".join([str(i) for i in x]) if isinstance(x, set) else "")
    
    print(f"合并后数据长度: {len(df_merge)}")
    print("数据加载完成")
    print(f"学生数量: {df_merge['UserId'].unique().size}")
    print(f"问题数量: {df_merge['QuestionId'].unique().size}")
    
    kcs = []
    for item in df_merge['SubjectId_level3'].values:
        if isinstance(item, set):
            kcs.extend(item)
    print(f"知识点数量: {len(set(kcs))}")
    
    return df_merge
def process_data(
    read_file: str,
    write_file: str,
    mode: int = 1,
    time_gap_weeks: int = 0,
    meta_data_dir=None, 
    task_name="task_3_4",
    test_ratio: float = 0.2,
    min_seq_len: int = 15,
    seed: int = 42,
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
        sample_size (int): 采样用户数，默认5000
    """
    set_seed(seed)
    stares = []

    # -------------------
    # 1. 读取、初步统计、清洗
    # -------------------
    # 加载数据
    df = load_nips_data(read_file, meta_data_dir, task_name)
    
    # 数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 数据清洗和预处理
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["UserId", "answer_timestamp", "SubjectId_level3_str", "IsCorrect", "QuestionId"])
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # 转换IsCorrect为数值类型
    df['IsCorrect'] = df['IsCorrect'].astype(int)
    
    # 记录「原始用户第一次出现」的顺序，用于之后排序
    orig_order = df["UserId"].astype(str).unique().tolist()

    # -------------------
    # 2. 模式1：保留每个用户对每题的第一次答题
    # -------------------
    if mode == 1:
        sorted_data = df.sort_values(
            by=["UserId", "QuestionId", "answer_timestamp", "tmp_index"]
        )
        processed_records = []
        seen = set()
        for _, row in sorted_data.iterrows():
            key = (row["UserId"], row["QuestionId"])
            if key not in seen:
                processed_records.append(row.to_dict())
                seen.add(key)
        processed_df = pd.DataFrame(processed_records)
    elif mode in [2, 3]:
        raise NotImplementedError(f"模式{mode}尚未实现，请使用模式1")
    else:
        raise ValueError(f"不支持的处理模式: {mode}")

    # 过滤掉序列长度<min_seq_len的学生
    student_seq_lens = processed_df.groupby('UserId').size()
    valid_students = student_seq_lens[student_seq_lens >= min_seq_len].index
    processed_df = processed_df[processed_df['UserId'].isin(valid_students)]
    print(f"过滤前学生数: {len(student_seq_lens)}, 过滤后学生数: {len(valid_students)}, 移除了 {len(student_seq_lens) - len(valid_students)} 名学生")
    
    # -------------------
    # 3. 分层抽取测试集
    # -------------------
    processed_df["is_test"] = False
    user_test_problems = {}

    for user_id in processed_df["UserId"].unique():
        user_block = processed_df[processed_df["UserId"] == user_id]
        corrects = user_block[user_block["IsCorrect"] == 1]["QuestionId"].unique()
        incorrects = user_block[user_block["IsCorrect"] == 0]["QuestionId"].unique()

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
            mask = (processed_df["UserId"] == user_id) & (
                processed_df["QuestionId"] == pid
            )
            processed_df.loc[mask, "is_test"] = True

        user_test_problems[str(user_id)] = set(test_set)

    # 打印测试集分布
    tc = processed_df["is_test"].sum()
    print(f"已标记 {tc} 个测试样本（{tc/len(processed_df):.2%}）")
    print(
        f"训练集正确率: {processed_df[~processed_df['is_test']]['IsCorrect'].mean():.4f}, "
        f"测试集正确率: {processed_df[processed_df['is_test']]['IsCorrect'].mean():.4f}"
    )

    # 保存测试集详情
    test_info_path = os.path.join(os.path.dirname(write_file), "test_problems.json")
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    with open(test_info_path, "w") as f:
        out = {
            k: [str(x) if isinstance(x, str) else (int(x) if hasattr(x, "item") else x) for x in sorted(list(v))]
            for k, v in user_test_problems.items()
        }
        json.dump(out, f, indent=2)

    # -------------------
    # 4. 时间间隔切分 & 合并（可选）
    # -------------------
    if time_gap_weeks > 0:
        before_cnt = processed_df["UserId"].nunique()
        processed_df, _ = split_sequences_by_time_gap(
            processed_df, time_gap_weeks, "UserId", "answer_timestamp"
        )
        processed_df, _ = improved_smart_sequence_merge(
            processed_df, time_gap_weeks, min_seq_len, "UserId", "answer_timestamp"
        )
        after_cnt = processed_df["UserId"].nunique()
        print(
            f"切分前用户数 {before_cnt}，切分后用户数 {after_cnt}；"
            f"测试样本仍为 {processed_df['is_test'].sum()}"
        )

    # 确保测试标记没丢
    assert processed_df["is_test"].sum() > 0, "错误：测试集标记丢失！"

    # -------------------
    # 5. 按「原始出现顺序 + 后缀」生成最终 UID 列表
    # -------------------
    processed_df["UserId"] = processed_df["UserId"].astype(str)
    all_ids = processed_df["UserId"].unique().tolist()

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
        block = processed_df[processed_df["UserId"] == uid].copy()
        # 按时间戳 + 原始 index 排序
        block = block.sort_values(by=["tmp_index"])
        seq_len = len(block)
        seq_problems = block["QuestionId"].tolist()
        seq_skills = block["SubjectId_level3_str"].tolist()
        seq_ans = block["IsCorrect"].astype(int).tolist()
        seq_times = block["answer_timestamp"].tolist()
        seq_costs = ["NA"] * seq_len
        
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
    print(f"数据处理完成，结果已保存至 {write_file}")


def read_data_from_csv(
    read_file: str,
    write_file: str,
    split_mode: int,
    time_gap_weeks: int = 0,
    meta_data_dir=None, 
    task_name="task_3_4", 
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
        meta_data_dir=meta_data_dir,
        task_name=task_name,
        test_ratio=test_ratio,
        min_seq_len=min_seq_len,
        seed=seed,
    )


