import pandas as pd
import numpy as np
import os
import json
import hashlib
from .utils import sta_infos, write_txt, format_list2str, set_seed

Q_FILE = "../data/math2/q.txt"
KEYS = ["user_id", "skill_id", "problem_id"]

def read_data_from_txt(math2_file, write_file, seed=42, test_ratio=0.2):
    """
    Processing method: Keep the original answer order, perform stratified sampling of the test set
    
    Args:
        math2_file (str): Input math data file path
        write_file (str): Output file path
        seed (int, optional): Random seed. Default is 42
        test_ratio (float, optional): Test set ratio. Default is 0.2
    """
    set_seed(seed)  # Set random seed
    stares = []
    
    # 读取数据
    math2_data = pd.read_csv(math2_file, sep='\t', header=None)
    q_data = pd.read_csv(Q_FILE, sep='\t', header=None)
    
    # 排除math2_data最后4列，q_data最后4行(主观题)
    math2_data = math2_data.iloc[:, :-4]
    q_data = q_data.iloc[:-4, :] 
    
    # 转换为DataFrame格式，保持原始顺序
    rows = []
    for student_id in range(len(math2_data)):
        # 保持每个学生的问题顺序
        for idx in range(math2_data.shape[1]):
            # 提取技能ID
            skill = '_'.join([str(i) for i, x in enumerate(q_data.iloc[idx]) if x == 1])
            rows.append({
                'user_id': str(student_id),
                'problem_id': str(idx),
                'skill_id': skill,
                'correct': str(math2_data.iloc[student_id, idx]),
                'order_id': str(idx)  # 使用原始顺序作为order_id
            })
    
    df = pd.DataFrame(rows)
    # 获取原始数据统计信息
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"interaction num: {ins}, user num: {us}, question num: {qs}, "
          f"concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # 由于每个学生只回答每个问题一次，不需要去重操作
    clean_df = df
    clean_df['correct'] = clean_df['correct'].astype(float)
    
    # ------------------
    # 分层抽取测试集
    # ------------------
    clean_df["is_test"] = False
    user_test_problems = {}
    
    for user_id in clean_df["user_id"].unique():
        user_block = clean_df[clean_df["user_id"] == user_id]
        corrects = user_block[user_block["correct"] == 1]["problem_id"].unique()
        incorrects = user_block[user_block["correct"] == 0]["problem_id"].unique()
        
        # 使用用户ID和种子生成一个唯一的哈希值用于随机状态
        h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2**32
        rng = np.random.RandomState(h ^ seed)
        
        # 计算要抽取的正确和错误问题数量
        n_corr = max(1, round(len(corrects) * test_ratio)) if len(corrects) > 0 else 0
        n_incorr = max(1, round(len(incorrects) * test_ratio)) if len(incorrects) > 0 else 0
        
        # 随机抽取测试问题
        sample_corr = rng.choice(corrects, size=n_corr, replace=False) if n_corr > 0 else []
        sample_incorr = rng.choice(incorrects, size=n_incorr, replace=False) if n_incorr > 0 else []
        
        # 合并正确和错误的测试样本
        test_set = np.concatenate([sample_corr, sample_incorr]) if (
            len(sample_corr) > 0 and len(sample_incorr) > 0
        ) else (sample_corr if len(sample_corr) > 0 else sample_incorr)
        
        # 标记测试集
        for pid in test_set:
            mask = (clean_df["user_id"] == user_id) & (clean_df["problem_id"] == pid)
            clean_df.loc[mask, "is_test"] = True
        
        # 记录测试问题
        user_test_problems[str(user_id)] = set(test_set)
    
    # 打印测试集分布
    tc = clean_df["is_test"].sum()
    print(f"Marked {tc} test samples ({tc/len(clean_df):.2%})")
    print(f"Training set accuracy: {clean_df[~clean_df['is_test']]['correct'].mean():.4f}, "
          f"Test set accuracy: {clean_df[clean_df['is_test']]['correct'].mean():.4f}")
    
    # 保存测试集详情
    # test_info_path = os.path.join(os.path.dirname(write_file), "test_problems.json")
    # os.makedirs(os.path.dirname(write_file), exist_ok=True)
    # with open(test_info_path, "w") as f:
    #     out = {
    #         k: [str(x) for x in sorted(list(v))]
    #         for k, v in user_test_problems.items()
    #     }
    #     json.dump(out, f, indent=2)
    
    # 按用户ID分组处理
    ui_df = clean_df.groupby('user_id', sort=False)
    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=['order_id'])  # 按原始顺序排序
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problem_id'].tolist()
        seq_skills = tmp_inter['skill_id'].tolist()
        seq_ans = tmp_inter['correct'].astype(int).tolist()
        seq_start_time = ['NA']
        seq_response_cost = ['NA']
        seq_is_test = tmp_inter['is_test'].astype(int).tolist()
        
        # 确保序列长度一致
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_is_test)
        
        user_inters.append(
            [[str(user), str(seq_len)], 
             format_list2str(seq_problems), 
             format_list2str(seq_skills), 
             format_list2str(seq_ans), 
             format_list2str(seq_start_time), 
             format_list2str(seq_response_cost),
             format_list2str(seq_is_test)])
    
    # 确保目录存在
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    if os.path.exists(write_file):
        os.remove(write_file)
        
    write_txt(write_file, user_inters)
    print(f"Data processing completed, results saved to {write_file}")
    # print(f"Test set information saved to {test_info_path}")
    print("\n".join(stares))
    return