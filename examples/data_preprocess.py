import os, sys
import argparse

from pycd.preprocess import data_proprocess, process_raw_data

from pycd.preprocess.split_datasets import main as split_dataset

# 数据集路径配置
dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "math1": "../data/math1/math1.txt",
    "math2": "../data/math2/math2.txt", 
    "frcsub": "../data/frcsub/frcsub.txt", 
    "junyi": "../data/junyi/junyi_ProblemLog_original.csv", 
    "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv", 
    "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv", 
    "ednet": "../data/ednet/",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "slp_math": "../data/slp_math/term-mat.csv",
    "jiuzhang": "../data/jiuzhang/practice_record_dump_0~2025.02.25_clean_add_info.csv",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv",
    "jiuzhang_g3": "../data/jiuzhang_g3/grade_3_data_en.csv",
    "jiuzhang_g4_g5": "../data/jiuzhang_g4_g5/grade_4_5_data.csv",
    "jiuzhang_g7": "../data/jiuzhang_g7/grade_7_data.csv"
}

# 配置文件路径
configf = "../configs/data_config.json"

if __name__ == "__main__":

    # 命令行参数设置
    
    parser = argparse.ArgumentParser(description='教育数据处理工具 - 从原始CSV到标准数据集')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    parser.add_argument('--min_seq_len', type=int, default=15, help='最小序列长度 (默认为15)')
    parser.add_argument('--split_mode', type=int,  default=1, help='题目多次回答处理策略,1保留第1次交互,2取学生在题目上多次的平均acc,3平均acc+所有学生在该题的acc')
    parser.add_argument('--time_info', type=int, default=0, help='是否考虑时间信息, 不考虑0, 考虑(1,2,4 week)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    print(args)

    dname, writef = process_raw_data(args.dataset_name, dname2paths, args.split_mode, args.time_info, args.test_ratio, args.min_seq_len)
    
    split_dataset(dname, writef, args.dataset_name, configf, args.min_seq_len, args.split_mode, args.time_info, args.test_ratio, args.n_folds, args.seed)