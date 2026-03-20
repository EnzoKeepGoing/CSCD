import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from .utils import set_seed, sta_infos, write_txt, format_list2str, split_sequences_by_time_gap

KEYS = ["student_id", "concept", "question_id"]

def check_equal(row):
    try:
        # 检查是否有非数字值
        if pd.isna(row['score']) or pd.isna(row['full_score']):
            return 0
        if row['score'] == 'n.a.' or row['full_score'] == 'n.a.':
            return 0
        
        # 转换为整数并比较
        score = int(float(row['score']))
        full_score = int(float(row['full_score']))
        return 1 if score == full_score else 0
    except Exception:
        # 任何转换错误都返回0
        return 0

def process_data(read_file, write_file, mode=1, time_gap_weeks=0, seed=42, alpha=0.5):
    """处理CSV数据并输出到TXT文件，支持三种不同的处理模式
    
    Args:
        read_file (str): 输入CSV文件路径
        write_file (str): 输出TXT文件路径
        mode (int): 处理模式 (1, 2, 或 3)
        time_gap_weeks (int): 时间间隔阈值(周)，大于0时会切分序列
        alpha (float): 模式3中的个人准确率权重系数
        seed (int): 随机种子
    """
    set_seed(seed)
    stares = []
    
    # 加载数据
    df = pd.read_csv(read_file, sep=',', header=0, index_col=False, encoding='utf-8', low_memory=False)
    
    # 数据统计
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {df.shape[0]}, user num: {df['student_id'].nunique()}, question num: {df['question_id'].nunique()}, "
        f"concept num: {df['concept'].nunique()}, avg(ins) per s:{avgins}, avg(c) per q:{avgcq}, na:{na}")

    # 分数转换
    df['correct'] = df.apply(check_equal, axis=1)
    df['concept'] = df['concept'].astype(str).str.replace(';', '_')

    # 数据清洗和预处理
    df["index"] = range(len(df))
    df = df.dropna(subset=["student_id", "question_id", "correct", "concept", "time_access"])
    df = df[df['concept'].astype(str).str.lower() != 'n.a.']
    df = df[df['correct'].isin([0, 1])]
    
    # 添加索引列并转换时间为毫秒
    df.loc[:, 'time_access'] = pd.to_datetime(df['time_access'])
    df['correct'] = df['correct'].astype(float)  # 确保可以计算平均值

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 基于时间间隔切分序列
    if time_gap_weeks > 0:
        df, gap_stats = split_sequences_by_time_gap(df, time_gap_weeks, 'student_id', 'time_access')
        print(f"时间间隔切分统计 (间隔={time_gap_weeks} 周):")
        for k, v in gap_stats.items():
            print(f"  {k}: {v}")
    
    # 预先为每个用户生成一致的问题随机顺序
    user_problem_orders = {}
    for user_id in df['student_id'].unique():
        # 获取该用户的所有唯一问题
        user_problems = df[df['student_id'] == user_id]['question_id'].unique()
        # 生成随机索引
        indices = np.arange(len(user_problems))
        np.random.shuffle(indices)
        # 创建问题到随机顺序的映射
        problem_order = {problem: order for problem, order in zip(user_problems, indices)}
        user_problem_orders[user_id] = problem_order  # {user:{question_id: index,xxx},xxx}
    
    # 根据不同模式处理数据
    if mode == 1:
        # 模式1: 保留每个用户对每个问题的第一次作答
        
        # 按用户ID、问题ID和回答顺序排序
        sorted_data = df.sort_values(by=['student_id', 'question_id', 'time_access', 'index'])
        
        # 初始化结果列表
        processed_records = []
        
        # 使用一个集合来跟踪已经处理过的用户-问题对
        processed_pairs = set()
        
        # 遍历排序后的数据
        for _, row in sorted_data.iterrows():
            user_id = row['student_id']
            problem_id = row['question_id']
            pair = (user_id, problem_id)
            
            # 如果这个用户-问题对还没处理过，保留它
            if pair not in processed_pairs:
                processed_records.append(row.to_dict())
                processed_pairs.add(pair)
        
        # 转换回DataFrame
        processed_df = pd.DataFrame(processed_records)
        
    elif mode == 2:
        # 模式2: 计算每个用户对每个问题的平均准确率
        
        # 使用defaultdict(list)收集每个用户-问题对的所有答案，然后计算平均值
        user_problem_answers = defaultdict(list)
        user_problem_skills = {}  # 存储每个用户-问题对的技能ID
        user_problem_timestamps = {}  # 存储最早的时间戳
        user_problem_response_times = defaultdict(list)  # 存储响应时间
        
        # 收集所有回答
        for _, row in df.iterrows():
            user_id = row['student_id']
            problem_id = row['question_id']
            correct = float(row['correct'])
            skill_id = row['concept']
            timestamp = row['time_access']
            
            pair = (user_id, problem_id)
            user_problem_answers[pair].append(correct)
            user_problem_skills[pair] = skill_id  # 假设技能ID对于同一问题是一致的(以最后一次为准)
            
            # 存储最早的时间戳
            if pair not in user_problem_timestamps or timestamp < user_problem_timestamps[pair]:
                user_problem_timestamps[pair] = timestamp
                
        
        # 计算平均值
        processed_records = []
        for (user_id, problem_id), answers in user_problem_answers.items():
            avg_correct = sum(answers) / len(answers)
            skill_id = user_problem_skills[(user_id, problem_id)]
            timestamp = user_problem_timestamps[(user_id, problem_id)]
            # avg_response_time = sum(user_problem_response_times[(user_id, problem_id)]) / len(user_problem_response_times[(user_id, problem_id)])
            
            processed_records.append({
                'student_id': user_id,
                'question_id': problem_id,
                'concept': skill_id,
                'correct': avg_correct,
                'time_access': timestamp,
                'response_time': 'NA',
                'attempt_count': len(answers)
            })
        
        # 转换回DataFrame
        processed_df = pd.DataFrame(processed_records)
        
        # 打印多次尝试的信息
        multi_attempts = [r for r in processed_records if r['attempt_count'] > 1]
        print(f"多次尝试的学生-问题对数量: {len(multi_attempts)}")
        
    elif mode == 3:
        # 模式3: 对多次作答使用加权平均
        
        # 第1步: 收集每个用户-问题对的所有回答
        user_problem_answers = defaultdict(list)
        user_problem_skills = {}  # 存储每个用户-问题对的技能ID
        user_problem_timestamps = {}  # 存储最早的时间戳
        
        for _, row in df.iterrows():
            user_id = row['student_id']
            problem_id = row['question_id']
            correct = float(row['correct'])
            skill_id = row['concept']
            timestamp = row['time_access']
            
            pair = (user_id, problem_id)
            user_problem_answers[pair].append(correct)
            user_problem_skills[pair] = skill_id
            
            # 存储最早的时间戳
            if pair not in user_problem_timestamps or timestamp < user_problem_timestamps[pair]:
                user_problem_timestamps[pair] = timestamp
                
            
        
        # 第2步: 计算每个问题的全局平均准确率
        problem_all_answers = defaultdict(list)
        for (user_id, problem_id), answers in user_problem_answers.items():
            for answer in answers:
                problem_all_answers[problem_id].append(answer)
        
        problem_global_avg = {}
        for problem_id, answers in problem_all_answers.items():
            problem_global_avg[problem_id] = sum(answers) / len(answers)
        
        # 第3步: 计算每个用户-问题对的加权平均
        processed_records = []
        for (user_id, problem_id), answers in user_problem_answers.items():
            # 计算个人平均准确率
            personal_avg = sum(answers) / len(answers)
            skill_id = user_problem_skills[(user_id, problem_id)]
            timestamp = user_problem_timestamps[(user_id, problem_id)]
            # avg_response_time = sum(user_problem_response_times[(user_id, problem_id)]) / len(user_problem_response_times[(user_id, problem_id)])
            
            # 如果有多次尝试，应用加权平均
            if len(answers) > 1:
                global_avg = problem_global_avg[problem_id]
                weighted_avg = alpha * personal_avg + (1 - alpha) * global_avg
            else:
                weighted_avg = personal_avg
            
            processed_records.append({
                'student_id': user_id,
                'question_id': problem_id,
                'concept': skill_id,
                'correct': weighted_avg,
                'time_access': timestamp,
                'response_time': 'NA',
                'attempt_count': len(answers)
            })
        
        # 转换回DataFrame
        processed_df = pd.DataFrame(processed_records)
        
        # 打印多次尝试的信息
        multi_attempts = [r for r in processed_records if r['attempt_count'] > 1]
        print(f"多次尝试的学生-问题对数量: {len(multi_attempts)}")
        print(f"加权平均的alpha值: {alpha}")
    
    # 处理用户交互数据 - 使用预先生成的随机顺序
    user_inters = []
    
    # 确保用户顺序一致 - 按用户ID排序以保持三种模式输出一致
    if time_gap_weeks > 0:
        # 当应用时间间隔切分时，使用字符串排序
        sorted_students = sorted(processed_df['student_id'].unique(), key=str)
    else:
        # 未应用时间间隔切分时，使用默认排序
        sorted_students = sorted(processed_df['student_id'].unique())
    
    for user_id in sorted_students:
        user_data = processed_df[processed_df['student_id'] == user_id].copy()
        
        # 获取该用户的问题列表和对应的数据
        problem_data = []
        for _, row in user_data.iterrows():
            problem_id = row['question_id']
            problem_data.append({
                'question_id': problem_id,
                'concept': row['concept'],
                'correct': row['correct'],
                'time_access': row['time_access'], 
                'response_time': 'NA',
                'random_order': user_problem_orders[user_id].get(problem_id, 0)
            })
        
        # 按随机顺序排序
        problem_data.sort(key=lambda x: x['random_order'])
        
        # 提取排序后的数据
        seq_len = len(problem_data)
        seq_problems = [item['question_id'] for item in problem_data]
        seq_skills = [item['concept'] for item in problem_data]
        
        # 根据模式选择是否保留小数
        if mode == 1:
            seq_ans = [round(item['correct']) for item in problem_data]  # 模式1使用整数
        else:
            seq_ans = [float(item['correct']) for item in problem_data]  # 模式2和3保留小数
            
        seq_submit_time = [item['time_access'] for item in problem_data]
        seq_response_cost = ['NA'] * seq_len  # 更新响应时间列表长度以匹配其他列表
        
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_submit_time)
        
        user_inters.append(
            [[str(user_id), str(seq_len)], 
             format_list2str(seq_problems), 
             format_list2str(seq_skills), 
             format_list2str(seq_ans), 
             format_list2str(seq_submit_time), 
             format_list2str(seq_response_cost)])
    
    # 确保目录存在
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    
    # 输出结果
    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return

def read_data_from_csv(read_file, write_file, split_mode, time_gap_weeks=0, seed=42):
    """处理CSV数据的主函数
    
    Args:
        read_file (str): 输入CSV文件路径
        write_file (str): 输出TXT文件路径
        split_mode (int): 处理模式 (1, 2, 或 3)
        time_gap_weeks (int): 时间间隔阈值(周)，用于切分长序列
        seed (int): 随机种子
    """
    if split_mode not in [1]:
        raise ValueError(f"SLP-math 不支持的处理模式: {split_mode}")
    
    # 调用处理函数
    process_data(read_file, write_file, mode=split_mode, time_gap_weeks=time_gap_weeks, seed=seed)