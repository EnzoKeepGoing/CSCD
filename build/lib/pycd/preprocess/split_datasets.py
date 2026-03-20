import os
import sys
import pandas as pd
import numpy as np
import json
import copy
from .utils import sta_infos, set_seed

def read_data(fname, min_seq_len=15, response_set=None, keep_all_sequences=True):
    """
    Read raw data, optional to delete user sequences with 
    length less than min_seq_len
    
    Args:
        fname: Data file path
        min_seq_len: Minimum sequence length
        response_set: Valid answer value set, if None accepts any value
        keep_all_sequences: Whether to keep all sequences, including short sequences
        
    Returns:
        DataFrame: Processed data frame
        set: Valid field set
    """
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    short_seqs = []  # Record short sequence information
    
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 7 == 0:  # stuid - Note: Now one user has 7 lines of data!
                effective_keys.add("uid")
                tmps = line.split(",")
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                
                # Check sequence length
                if seq_len < min_seq_len:
                    if not keep_all_sequences:
                        # Original logic: Skip short sequences
                        i += 7
                        dcur = dict()
                        delstu += 1
                        delnum += seq_len
                        continue
                    else:
                        # Record short sequences, but do not skip
                        short_seqs.append({"uid": stuid, "length": seq_len})
                
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 7 == 1:  # question ids
                qs = []
                if line.find("NA") == -1:
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 7 == 2:  # concept ids
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 7 == 3:  # responses
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            # Try to convert response to float to support decimal values
                            r_value = float(r)
                            
                            # Only check if a response_set is specified
                            if response_set is not None and r_value not in response_set:
                                print(f"Warning: The answer value {r_value} in line {i} is not in the specified valid value set")
                                # Note: Processing continues, just发出警告
                            
                            rs.append(r_value)
                        except ValueError:
                            print(f"Error: The answer value '{r}' in line {i} is not a valid number")
                            flag = False
                            break
                    
                    if not flag:
                        i += 4  # Jump to the next user
                        dcur = dict()
                        badr += 1
                        continue
                
                dcur["responses"] = rs
            elif i % 7 == 4:  # timestamps
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 7 == 5:  # usets
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets
            elif i % 7 == 6:  # is_test标记 - 新增的测试集标记行
                effective_keys.add("is_test")
                is_test = []
                if line.find("NA") == -1:
                    is_test = line.split(",")
                dcur["is_test"] = is_test

                # After processing all 7 lines of data for a user, add the data to the result
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        # 确保数值正确转换为字符串
                        if key == "responses":
                            dres[key].append(",".join([str(float(k)) for k in dcur[key]]))
                        else:
                            dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    
    df = pd.DataFrame(dres)
    
    # 输出统计信息
    if keep_all_sequences and short_seqs:
        print(f"Kept {len(short_seqs)} short sequences with length less than {min_seq_len}, the shortest length is {min([s['length'] for s in short_seqs])}")
    else:
        print(f"Deleted {delstu} students with sequences less than {min_seq_len}, deleted {delnum} interactions")
    
    print(f"Invalid answer number: {badr}, valid interaction number: {goodnum}")
    
    return df, effective_keys

def build_item_matrix(df, save_path):
    """
    Build item-concept matrix and save as txt file, using original id values)
    
    Args:
        df: Data frame containing questions and concepts
        save_path: Save path
    """
    q_kc_dict = {}
    
    for _, row in df.iterrows():
        questions = row["questions"].split(",")
        concepts = row["concepts"].split(",")
        
        for q, c in zip(questions, concepts):
            if q == "NA" or c == "NA":
                continue
            
            if q not in q_kc_dict:
                q_kc_dict[q] = []  # Use list instead of set to maintain order
            
            # 处理多概念情况
            if "_" in c:
                concepts_list = c.split("_")
                for concept in concepts_list:
                    # 只添加不重复的概念
                    if concept not in q_kc_dict[q]:
                        q_kc_dict[q].append(concept)
            else:
                if c not in q_kc_dict[q]:
                    q_kc_dict[q].append(c)
    
    # 写入txt文件
    # with open(save_path, "w", encoding="utf8") as f:
    #     for q, kcs in q_kc_dict.items():
    #         f.write(f"{q}:{','.join(kcs)}\n")
    
    # print(f"问题-知识点对应关系已保存到 {save_path}, 包含 {len(q_kc_dict)} 个问题")
    return q_kc_dict

def id_mapping(df):
    """
    Map original IDs to consecutive index values and ensure each ID is correctly mapped
    """
    id_keys = ["questions", "concepts", "uid"]
    dkeyid2idx = dict()
    
    # 处理所有ID类型
    for key in id_keys:
        if key not in df.columns:
            continue
            
        dkeyid2idx.setdefault(key, dict())
        
        if key == "concepts":
            # Special processing for concept IDs, including splitting compound concepts
            all_concept_ids = set()
            for _, row in df.iterrows():
                for c in row[key].split(","):
                    # 先添加原始概念ID（不再需要）
                    # all_concept_ids.add(c)
                    # 拆分复合概念ID
                    if "_" in c:
                        for sub_c in c.split("_"):
                            all_concept_ids.add(sub_c)
                    else:
                        all_concept_ids.add(c)
            
            # 为所有概念ID创建映射
            for c in sorted(all_concept_ids):
                if c not in dkeyid2idx[key]:
                    dkeyid2idx[key][c] = len(dkeyid2idx[key])
        else:
            # 处理其他ID类型
            for _, row in df.iterrows():
                if key == "uid":
                    # uid列是直接值而不是逗号分隔列表
                    id_val = row[key]
                    if id_val not in dkeyid2idx[key]:
                        dkeyid2idx[key][id_val] = len(dkeyid2idx[key])
                else:
                    # 处理逗号分隔的ID列表
                    for id_val in row[key].split(","):
                        if id_val not in dkeyid2idx[key]:
                            dkeyid2idx[key][id_val] = len(dkeyid2idx[key])
    
    return dkeyid2idx

def split_user_sequence_cv(df, id_maps=None, n_folds=5, random_seed=42):
    """
    Split user sequences for cross-validation:
    1. Use existing test set marks (is_test) to split the test set
    2. Split non-test data based on answer accuracy for stratified cross-validation
    
    Args:
        df: Data frame containing user sequences and test set marks
        id_maps: ID mapping dictionary
        n_folds: Number of folds for cross-validation
        random_seed: Random seed
        
    Returns:
        DataFrame: Training/validation data frame (with fold column)
        DataFrame: Test set data frame
    """
    # 设置随机种子
    set_seed(random_seed)
    
    # 初始化数据结构
    train_valid_rows = []
    test_rows = []
    
    print(f"Data splitting: Using existing test set marks, adding {n_folds} folds based on accuracy for stratified cross-validation")
    
    # 检查是否有测试集标记列
    if "is_test" not in df.columns:
        print("Warning: No test set mark column in data, all data will be used as training set")
    
    # 处理每个用户的序列数据
    for _, row in df.iterrows():
        user_id = row["uid"]
        questions = row["questions"].split(",")
        responses = row["responses"].split(",")
        
        # 获取测试集标记
        is_test_flags = []
        if "is_test" in df.columns and not pd.isna(row["is_test"]):
            is_test_flags = [int(t) for t in row["is_test"].split(",")]
        
        # 确保测试集标记与问题数量匹配
        if len(is_test_flags) != len(questions):
            print(f"Warning: The test set mark length ({len(is_test_flags)}) of user {user_id} does not match the number of questions ({len(questions)})")
            # 填充或截断测试集标记
            is_test_flags = is_test_flags[:len(questions)] if len(is_test_flags) > len(questions) else is_test_flags + [0] * (len(questions) - len(is_test_flags))
        
        # 处理每个问题
        for i in range(len(questions)):
            q_id = questions[i]
            correct = float(responses[i])
            
            # 确定是否为测试集
            is_test = (i < len(is_test_flags) and is_test_flags[i] == 1)
            
            # 创建数据行（使用ID映射或原始ID）
            if id_maps:
                mapped_user_id = str(id_maps["uid"][user_id])
                mapped_q_id = str(id_maps["questions"][q_id])
                new_row = {
                    "user_id": mapped_user_id,
                    "question_id": mapped_q_id,
                    "correct": correct
                }
            else:
                new_row = {
                    "user_id": user_id,
                    "question_id": q_id,
                    "correct": correct
                }
            
            # 根据测试集标记分配
            if is_test:
                # 对于测试集数据，标记fold为-1
                new_row["fold"] = -1
                test_rows.append(new_row)
            else:
                # 先不分配fold，只添加到训练/验证集
                train_valid_rows.append(new_row)
    
    # 基于正确率对非测试集数据进行分层
    if train_valid_rows:
        # 将训练/验证数据按正确率分组
        correct_records = [row for row in train_valid_rows if row["correct"] == 1.0]
        incorrect_records = [row for row in train_valid_rows if row["correct"] == 0.0]
        
        # 设置随机种子确保可重现性
        np.random.seed(random_seed)
        
        # 随机打乱两组数据
        np.random.shuffle(correct_records)
        np.random.shuffle(incorrect_records)
        
        # 计算每个fold应包含的记录数（向上取整以确保所有数据都被分配）
        n_correct_per_fold = (len(correct_records) + n_folds - 1) // n_folds
        n_incorrect_per_fold = (len(incorrect_records) + n_folds - 1) // n_folds
        
        # 分配fold
        for i, record in enumerate(correct_records):
            fold = min(i // n_correct_per_fold, n_folds - 1)
            record["fold"] = fold
        
        for i, record in enumerate(incorrect_records):
            fold = min(i // n_incorrect_per_fold, n_folds - 1)
            record["fold"] = fold
        
        # 合并分层后的记录
        train_valid_rows = correct_records + incorrect_records
    
    # 转换为DataFrame
    train_valid_df = pd.DataFrame(train_valid_rows)
    test_df = pd.DataFrame(test_rows)
    
    # 计算统计信息
    test_count = len(test_df)
    train_valid_count = len(train_valid_df)
    
    print(f"Test set size: {test_count} records")
    print(f"Training/validation set size: {train_valid_count} records")
    
    # 计算正确率统计
    if not test_df.empty:
        test_acc = test_df["correct"].mean()
        print(f"Test set accuracy: {test_acc:.4f}")
    
    if not train_valid_df.empty:
        train_valid_acc = train_valid_df["correct"].mean()
        print(f"Training/validation set accuracy: {train_valid_acc:.4f}")
        
        # 打印每个fold的统计信息
        for fold in range(n_folds):
            fold_data = train_valid_df[train_valid_df["fold"] == fold]
            fold_count = len(fold_data)
            if fold_count > 0:
                fold_acc = fold_data["correct"].mean()
                print(f"Fold {fold}: {fold_count} records, accuracy: {fold_acc:.4f}")
            else:
                print(f"Fold {fold}: 0 records")
    
    return train_valid_df, test_df

def create_q_matrix_file(df, id_maps, save_path):
    """
    Create q_matrix.csv file, using correct quote format to represent concept lists
    Write directly to CSV to avoid pandas' quote escaping behavior
    """
    # 构建问题-概念映射
    q_kc_dict = {}
    
    for _, row in df.iterrows():
        questions = row["questions"].split(",")
        concepts = row["concepts"].split(",")
        
        for q, c in zip(questions, concepts):
            if q not in q_kc_dict:
                q_kc_dict[q] = set()
            
            # 处理多概念情况
            if "_" in c:
                sub_concepts = c.split("_")
                for sub_c in sub_concepts:
                    q_kc_dict[q].add(sub_c)
            else:
                q_kc_dict[q].add(c)
    
    # 直接写入CSV文件
    with open(save_path, 'w', newline='') as f:
        f.write("question_id,concept_ids\n")  # Write title line
        
        # 处理每个问题-概念对
        for q_orig, kcs in q_kc_dict.items():
            if q_orig not in id_maps["questions"]:
                print(f"Warning: Question ID {q_orig} is not in the mapping, skipping")
                continue
                
            # 映射问题ID
            q_idx = str(id_maps["questions"][q_orig])
            
            # 映射概念ID
            mapped_kcs = []
            for kc in kcs:
                if kc in id_maps["concepts"]:
                    mapped_kcs.append(str(id_maps["concepts"][kc]))
                else:
                    print(f"Warning: Concept ID {kc} is not in the mapping, skipping")
            
            if not mapped_kcs:
                print(f"Warning: Question ID {q_orig} has no valid concept ID, using default concept ID 0")
                f.write(f"{q_idx},[0]\n")
                continue
            
            # Use square brackets format to represent concept list
            if len(mapped_kcs) > 1:
                # Multiple concepts use square brackets and commas, only use one pair of double quotes
                kc_str = f"[{', '.join(mapped_kcs)}]"
                f.write(f'{q_idx},"{kc_str}"\n')
            else:
                # Single concept also uses square brackets, no quotes needed
                f.write(f"{q_idx},[{mapped_kcs[0]}]\n")
    
    print(f"Q-matrix file has been saved to {save_path}")
    
    return pd.read_csv(save_path)

def calculate_statistics_cv(train_valid_df, test_df, item_df, n_folds):
    """
    Calculate statistics for cross-validation dataset
    
    Args:
        train_valid_df: Training/validation data frame (with fold column)
        test_df: Test set data frame
        item_df: Question-concept mapping data frame
        n_folds: Number of folds for cross-validation
        
    Returns:
        dict: Statistics dictionary
    """
    # 初始化统计信息字典
    stats = {
        "users": {"test": test_df["user_id"].nunique()},
        "questions": {"test": test_df["question_id"].nunique()},
        "interactions": {"test": test_df.shape[0]},
        "folds": {}
    }
    
    # 计算总体统计信息
    stats["users"]["train_valid"] = train_valid_df["user_id"].nunique()
    stats["questions"]["train_valid"] = train_valid_df["question_id"].nunique()
    stats["interactions"]["train_valid"] = train_valid_df.shape[0]
    
    # 计算每个fold的统计信息
    for fold in range(n_folds):
        fold_df = train_valid_df[train_valid_df["fold"] == fold]
        other_folds_df = train_valid_df[train_valid_df["fold"] != fold]
        
        stats["folds"][fold] = {
            "validation": {
                "users": fold_df["user_id"].nunique(),
                "questions": fold_df["question_id"].nunique(),
                "interactions": fold_df.shape[0]
            },
            "training": {
                "users": other_folds_df["user_id"].nunique(),
                "questions": other_folds_df["question_id"].nunique(),
                "interactions": other_folds_df.shape[0]
            }
        }
    
    # 计算总的用户、问题和交互数
    stats["users"]["total"] = len(set(test_df["user_id"].tolist() + train_valid_df["user_id"].tolist()))
    stats["questions"]["total"] = len(set(test_df["question_id"].tolist() + train_valid_df["question_id"].tolist()))
    stats["interactions"]["total"] = test_df.shape[0] + train_valid_df.shape[0]
    
    # 计算唯一概念数量
    all_concepts = set()
    for concept_str in item_df["concept_ids"]:
        # 去除可能的引号和方括号，提取纯数字
        cleaned = concept_str.replace('"', '').replace('[', '').replace(']', '')
        for c in cleaned.split(','):
            c = c.strip()
            if c:  # 确保不是空字符串
                all_concepts.add(c)
    
    stats["concepts"] = {"total": len(all_concepts)}
    
    # 打印统计信息
    print("\n===== Dataset statistics =====")
    print(f"Total number of users: {stats['users']['total']}")
    print(f"Total number of questions: {stats['questions']['total']}")
    print(f"Total number of interactions: {stats['interactions']['total']}")
    print(f"Total number of concepts: {stats['concepts']['total']}")
    
    print(f"\nTraining/validation set: {stats['users']['train_valid']}, "
          f"questions {stats['questions']['train_valid']}, "
          f"interactions {stats['interactions']['train_valid']}")
    print(f"Test set: {stats['users']['test']}, "
          f"questions {stats['questions']['test']}, "
          f"interactions {stats['interactions']['test']}")
    
    print("\nEach fold statistics:")
    for fold in range(n_folds):
        print(f"Fold {fold}:")
        print(f"   Validation set: {stats['folds'][fold]['validation']['users']}, "
              f"questions {stats['folds'][fold]['validation']['questions']}, "
              f"interactions {stats['folds'][fold]['validation']['interactions']}")
        print(f"   Training set: {stats['folds'][fold]['training']['users']}, "
              f"questions {stats['folds'][fold]['training']['questions']}, "
              f"interactions {stats['folds'][fold]['training']['interactions']}")
    
    # 计算平均正确率
    test_acc = test_df["correct"].astype(float).mean() * 100
    train_valid_acc = train_valid_df["correct"].astype(float).mean() * 100
    
    print(f"\nAccuracy: Training/validation set {train_valid_acc:.2f}%, Test set {test_acc:.2f}%")
    
    return stats


def write_config_cv(config_path, dataset_name, stats, id_maps, data_path, n_folds=5, split_mode=1, time_info=0, min_seq_len=15):
    """
    Write configuration information for cross-validation dataset to configuration file
    
    Args:
        config_path: Configuration file path
        dataset_name: Dataset name
        stats: Statistics
        id_maps: ID mapping information
        data_path: Data saving path
        n_folds: Number of cross-validation folds
        split_mode: Data set splitting mode
    """
    folds = list(range(0, n_folds))
    config = {
        "data_path": data_path,
        "num_users": stats["users"]["total"],
        "num_questions": stats["questions"]["total"],
        "num_concepts": stats["concepts"]["total"],
        "num_interactions": stats["interactions"]["total"],
        "train_valid_file": "train_valid.csv",
        "test_file": "test.csv",
        "q_matrix_file": "q_matrix.csv",
        "id_mapping_file": "id_mapping.json",
        "min_seq_len": min_seq_len,
        "split_mode": split_mode,
        "time_info": time_info,
        "folds": folds
    }
    
    # 读取现有配置或创建新的
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            content = f.read().strip()
            if content:
                all_configs = json.loads(content)
            else:
                all_configs = {}
    else:
        all_configs = {}
    
    # 更新配置
    all_configs[dataset_name] = config
    
    # 写入配置文件
    with open(config_path, "w") as f:
        json.dump(all_configs, f, indent=4)
    
    print(f"Dataset configuration information saved to {config_path}")

def main(dname, fname, dataset_name, config_path, min_seq_len=15, split_mode=1, time_info=0, test_ratio=0.2, n_folds=5, random_seed=42):
    """
    Main function to process dataset
    """
    # Create directory
    os.makedirs(dname, exist_ok=True)
    
    # 1. 读取原始数据，删除短序列
    df, effective_keys = read_data(fname, min_seq_len)
    
    # 检查必要的字段
    if "questions" not in effective_keys or "concepts" not in effective_keys or "responses" not in effective_keys:
        print("Error: Data missing necessary fields (questions, concepts, responses)")
        return
    
    # 2. 构建Q矩阵-映射前id值
    q_matrix_path = os.path.join(dname, "que_kc_lookup.txt")
    q_kc_dict = build_item_matrix(df, q_matrix_path)
    
    # 3. 创建ID映射字典
    id_maps = id_mapping(df)
    
    # 保存ID映射
    id_map_path = os.path.join(dname, "id_mapping.json")
    with open(id_map_path, "w") as f:
        json.dump(id_maps, f, indent=4)
    print(f"ID mapping has been saved to {id_map_path}")
    
    # 4. 数据集划分 - 使用映射后的ID
    train_valid_df, test_df = split_user_sequence_cv(df, id_maps, n_folds, random_seed)
    
    # 5. 保存数据集
    train_valid_df.to_csv(os.path.join(dname, "train_valid.csv"), index=False)
    test_df.to_csv(os.path.join(dname, "test.csv"), index=False)
    
    # 6. 创建Q矩阵-映射后id值
    item_path = os.path.join(dname, "q_matrix.csv")
    item_df = create_q_matrix_file(df, id_maps, item_path)
    
    # 7. 计算统计信息
    stats = calculate_statistics_cv(train_valid_df, test_df, item_df, n_folds)
    
    # 8. 写入配置文件
    write_config_cv(config_path, dataset_name, stats, id_maps, dname, n_folds, split_mode, time_info, min_seq_len)
    
    
    print(f"\nData processing completed! Dataset {dataset_name} has been saved to {dname}")

# 主函数调用示例
if __name__ == "__main__":
   
    main(data_dir, input_file, dataset_name, config_file, min_seq_len, split_mode, test_ratio, n_folds, random_seed)