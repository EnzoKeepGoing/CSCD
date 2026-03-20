import pandas as pd


def set_seed(seed):
    """设置全局随机种子。
    
    Args:
        seed (int): 随机种子
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda环境变量设置
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        # print(df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]])
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in str(ks).split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def replace_text(text):
    text = text.replace("_", "####").replace(",", "@@@@")
    return text


def format_list2str(input_list):
    return [str(x) for x in input_list]


def one_row_concept_to_question(row):
    """Convert one row from concept to question

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_question = []
    new_concept = []
    new_response = []

    tmp_concept = []
    begin = True
    for q, c, r, mask, is_repeat in zip(row['questions'].split(","),
                                        row['concepts'].split(","),
                                        row['responses'].split(","),
                                        row['selectmasks'].split(","),
                                        row['is_repeat'].split(","),
                                        ):
        if begin:
            is_repeat = "0"
            begin = False
        if mask == '-1':
            break
        if is_repeat == "0":
            if len(tmp_concept) != 0:
                new_concept.append("_".join(tmp_concept))
                tmp_concept = []
            new_question.append(q)
            new_response.append(r)
            tmp_concept = [c]
        else:#如果是 1 就累计知识点
            tmp_concept.append(c)
    if len(tmp_concept) != 0:
        new_concept.append("_".join(tmp_concept))

    if len(new_question) < 200:
        pads = ['-1'] * (200 - len(new_question))
        new_question += pads
        new_concept += pads
        new_response += pads

    new_selectmask = ['1']*len(new_question)
    new_is_repeat = ['0']*len(new_question)

    new_row = {"fold": row['fold'],
               "uid": row['uid'],
               "questions": ','.join(new_question),
               "concepts": ','.join(new_concept),
               "responses": ','.join(new_response),
               "selectmasks": ','.join(new_selectmask),
               "is_repeat": ','.join(new_is_repeat),
               }
    return new_row

def concept_to_question(df):
    """Convert df from concept to question
    Args:
        df (_type_): df contains concept

    Returns:
        _type_: df contains question
    """
    new_row_list = list(df.apply(one_row_concept_to_question,axis=1).values)
    df_new = pd.DataFrame(new_row_list)
    return df_new

def get_df_from_row(row):
    value_dict = {}
    for col in ['questions', 'concepts', 'responses', 'is_repeat']:
        value_dict[col] = row[col].split(",")
    df_value = pd.DataFrame(value_dict)
    df_value = df_value[df_value['questions']!='-1']
    return df_value



def split_sequences_by_time_gap(df, time_gap_weeks=0, 
                                        user_id_col='studentId', timestamp_col='startTime', 
                                        timestamp_unit='ms'):
    """基于时间间隔切分学生序列，考虑相邻交互间隔和从序列开始的累积时间
    
    Args:
        df (DataFrame): 原始数据框
        time_gap_weeks (int): 时间间隔阈值，单位为周
        user_id_col (str): 用户ID列名
        timestamp_col (str): 时间戳列名
        timestamp_unit (str): 时间戳单位
        
    Returns:
        DataFrame: 切分后的数据框
        dict: 统计信息
    """
    if time_gap_weeks <= 0:
        # 不进行切分，保持原样
        return df, {
            "original_students": df[user_id_col].nunique(),
            "virtual_students": df[user_id_col].nunique(),
            "split_count": 0,
            "max_splits_for_one_student": 0,
            "avg_sequence_length_before": df.groupby(user_id_col).size().mean(),
            "avg_sequence_length_after": df.groupby(user_id_col).size().mean()
        }
    
    # 时间间隔阈值转换
    time_gap_seconds = time_gap_weeks * 7 * 24 * 60 * 60  # 转换为秒
    
    # 根据时间戳单位确定阈值
    if timestamp_unit == 'ms':
        time_gap_threshold = time_gap_seconds * 1000  # 毫秒
    elif timestamp_unit == 's':
        time_gap_threshold = time_gap_seconds  # 秒
    else:
        raise ValueError(f"不支持的时间戳单位: {timestamp_unit}，请使用 'ms' 或 's'")
    
    # 按用户ID分组处理
    all_new_rows = []
    virtual_student_counter = 0
    max_splits = 0
    splits_by_student = {}
    
    # 记录原始序列长度
    orig_seq_lengths = df.groupby(user_id_col).size()
    
    for user_id, user_data in df.groupby(user_id_col):
        # 按时间戳排序
        user_data = user_data.sort_values(timestamp_col)
        timestamps = user_data[timestamp_col].values
        
        # 检查时间间隔条件
        split_indices = []
        sequence_start_time = timestamps[0]  # 当前序列的起始时间
        
        for i in range(1, len(timestamps)):
            # 检查两个条件
            adjacent_gap = timestamps[i] - timestamps[i-1]  # 相邻交互时间差
            accumulated_duration = timestamps[i] - sequence_start_time  # 从当前序列开始的累积时间
            
            # 如果任一条件满足，创建分割点
            if adjacent_gap > time_gap_threshold or accumulated_duration > time_gap_threshold:
                split_indices.append(i)
                sequence_start_time = timestamps[i]  # 重置序列起始时间
        
        # 记录分割点数量
        splits_by_student[user_id] = len(split_indices)
        max_splits = max(max_splits, len(split_indices))
        
        # 执行切分逻辑
        if not split_indices:  # 没有分割点
            all_new_rows.append(user_data)
        else:
            # 生成虚拟学生ID
            virtual_ids = [f"{user_id}_{j+1}" for j in range(len(split_indices) + 1)]
            
            # 切分序列
            start_idx = 0
            for seq_idx, end_idx in enumerate(split_indices):
                segment = user_data.iloc[start_idx:end_idx].copy()
                segment[user_id_col] = virtual_ids[seq_idx]
                all_new_rows.append(segment)
                start_idx = end_idx
            
            # 添加最后一个分段
            last_segment = user_data.iloc[start_idx:].copy()
            last_segment[user_id_col] = virtual_ids[-1]
            all_new_rows.append(last_segment)
            
            virtual_student_counter += len(virtual_ids) - 1
    
    # 合并所有切分后的数据
    result_df = pd.concat(all_new_rows, ignore_index=True)
    
    # 计算切分后的序列长度
    new_seq_lengths = result_df.groupby(user_id_col).size()
    
    # 统计信息
    stats = {
        "original_students": df[user_id_col].nunique(),
        "virtual_students": result_df[user_id_col].nunique(),
        "new_students_created": virtual_student_counter,
        "split_count": sum(splits_by_student.values()),
        "max_splits_for_one_student": max_splits,
        "avg_sequence_length_before": orig_seq_lengths.mean(),
        "avg_sequence_length_after": new_seq_lengths.mean()
    }
    
    return result_df, stats


def improved_smart_sequence_merge(df, time_gap_weeks, min_seq_len, user_id_col='studentId', timestamp_col='startTime'):
    """改进的智能序列合并方法，按照时间接近度合并短序列"""
    
    # 1. 首先进行标准的时间间隔切分
    split_df, stats = split_sequences_by_time_gap(df, time_gap_weeks, user_id_col, timestamp_col)
    
    # 如果不需要切分，直接返回
    if time_gap_weeks <= 0:
        return split_df, stats
    
    # 2. 按原始学生ID分组处理
    result_sequences = []
    merge_operations = 0
    
    # 按原始学生ID分组
    original_student_groups = {}
    for virtual_id in split_df[user_id_col].unique():
        virtual_seq = split_df[split_df[user_id_col] == virtual_id].copy()
        
        # 获取原始学生ID
        if '_' in str(virtual_id):
            original_id = str(virtual_id).split('_')[0]
        else:
            original_id = virtual_id
            
        if original_id not in original_student_groups:
            original_student_groups[original_id] = []
        
        original_student_groups[original_id].append((virtual_id, virtual_seq))
    
    # 3. 处理每个原始学生的所有序列
    for original_id, virtual_seqs in original_student_groups.items():
        # 划分长短序列
        long_seqs = []  # 长度>=min_seq_len的序列
        short_seqs = []  # 长度<min_seq_len的序列
        
        for virtual_id, seq in virtual_seqs:
            if len(seq) >= min_seq_len:
                long_seqs.append((virtual_id, seq))
            else:
                short_seqs.append((virtual_id, seq))
        
        # 如果没有短序列，直接保留所有长序列
        if not short_seqs:
            for _, seq in long_seqs:
                result_sequences.append(seq)
            continue
        
        # 如果没有长序列，则开始合并短序列
        if not long_seqs:
            # 当没有长序列时，合并短序列
            merged_results = merge_short_sequences(short_seqs, original_id, min_seq_len, timestamp_col, user_id_col)
            result_sequences.extend(merged_results)
            merge_operations += len(short_seqs) - len(merged_results)
        else:
            # 有长序列，将短序列合并到最接近的长序列
            processed_long_seqs = {}
            
            # 先保存所有长序列
            for long_id, long_seq in long_seqs:
                processed_long_seqs[long_id] = long_seq.copy()
            
            # 处理每个短序列
            for short_id, short_seq in short_seqs:
                # 找到时间上最接近的长序列
                closest_long_id = None
                min_time_gap = float('inf')
                
                for long_id, long_seq in processed_long_seqs.items():
                    time_gap = calculate_time_gap(short_seq, long_seq, timestamp_col)
                    if time_gap < min_time_gap:
                        min_time_gap = time_gap
                        closest_long_id = long_id
                
                # 将短序列合并到最接近的长序列
                if closest_long_id is not None:
                    # 合并序列
                    processed_long_seqs[closest_long_id] = pd.concat(
                        [processed_long_seqs[closest_long_id], short_seq]
                    ).sort_values(timestamp_col)
                    
                    # 更新ID
                    processed_long_seqs[closest_long_id][user_id_col] = closest_long_id
                    merge_operations += 1
            
            # 添加处理后的长序列
            for _, seq in processed_long_seqs.items():
                result_sequences.append(seq)
    
    # 4. 合并所有结果序列
    final_df = pd.concat(result_sequences, ignore_index=True)
    
    # 更新统计信息
    updated_stats = {
        "original_students": stats["original_students"],
        "virtual_students_before_merge": stats["virtual_students"],
        "virtual_students_after_merge": final_df[user_id_col].nunique(),
        "merge_operations": merge_operations,
        "avg_sequence_length_before": stats["avg_sequence_length_before"],
        "avg_sequence_length_after": final_df.groupby(user_id_col).size().mean()
    }
    
    return final_df, updated_stats

def merge_short_sequences(short_seqs, original_id, min_seq_len, timestamp_col, user_id_col='studentId'):
    """合并短序列，直到达到最小长度"""
    result_sequences = []
    merge_operations = 0
    
    # 如果没有序列，直接返回
    if not short_seqs:
        return result_sequences
    
    # 复制序列列表，以便修改
    remaining_seqs = short_seqs.copy()
    
    # 循环直到处理完所有序列
    while remaining_seqs:
        # 从第一个序列开始
        current_id, current_seq = remaining_seqs.pop(0)
        current_seq = current_seq.copy()
        merge_count = 1
        
        # 不断合并，直到达到最小长度或没有更多序列
        while len(current_seq) < min_seq_len and remaining_seqs:
            # 找到时间上最接近的序列
            closest_idx, _ = find_closest_sequence(current_seq, remaining_seqs, timestamp_col)
            
            # 合并
            next_id, next_seq = remaining_seqs.pop(closest_idx)
            current_seq = pd.concat([current_seq, next_seq]).sort_values(timestamp_col)
            merge_count += 1
            
            # 更新ID为合并格式
            current_id = f"{original_id}_merged_{len(result_sequences) + 1}"
            current_seq.loc[:, user_id_col] = current_id
        
        # 添加合并后的序列
        result_sequences.append(current_seq)
    
    return result_sequences

def find_closest_sequence(reference_seq, candidate_seqs, timestamp_col):
    """找到时间上最接近的序列"""
    reference_start = reference_seq[timestamp_col].min()
    reference_end = reference_seq[timestamp_col].max()
    
    min_gap = float('inf')
    closest_idx = 0
    
    for i, (_, candidate) in enumerate(candidate_seqs):
        candidate_start = candidate[timestamp_col].min()
        candidate_end = candidate[timestamp_col].max()
        
        # 计算时间差
        if candidate_end < reference_start:
            # 候选序列在参考序列之前
            gap = reference_start - candidate_end
        elif candidate_start > reference_end:
            # 候选序列在参考序列之后
            gap = candidate_start - reference_end
        else:
            # 序列有重叠
            gap = 0
            
        if gap < min_gap:
            min_gap = gap
            closest_idx = i
    
    return closest_idx, min_gap

def calculate_time_gap(seq1, seq2, timestamp_col):
    """计算两个序列之间的时间差"""
    seq1_start = seq1[timestamp_col].min()
    seq1_end = seq1[timestamp_col].max()
    seq2_start = seq2[timestamp_col].min()
    seq2_end = seq2[timestamp_col].max()
    
    if seq1_end < seq2_start:
        # seq1完全在seq2之前
        return seq2_start - seq1_end
    elif seq2_end < seq1_start:
        # seq2完全在seq1之前
        return seq1_start - seq2_end
    else:
        # 序列有重叠
        return 0


