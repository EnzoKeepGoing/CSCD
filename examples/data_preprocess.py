import os, sys
import argparse

from pycd.preprocess import data_proprocess, process_raw_data
from pycd.preprocess.split_datasets import main as split_dataset

# Dataset path configuration
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

# Configuration file path
configf = "../configs/data_config.json"

if __name__ == "__main__":

    # Command-line arguments
    
    parser = argparse.ArgumentParser(
        description='Educational Data Processing Tool - From Raw CSV to Standardized Dataset'
    )
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--min_seq_len', type=int, default=15, help='Minimum sequence length (default: 15)')
    parser.add_argument(
        '--split_mode',
        type=int,
        default=1,
        help='Strategy for handling multiple attempts on the same question: '
             '1 = keep first attempt; '
             '2 = average accuracy per student-question; '
             '3 = average accuracy + global question accuracy'
    )
    parser.add_argument(
        '--time_info',
        type=int,
        default=0,
        help='Whether to include temporal information: '
             '0 = no; '
             '1/2/4 = time windows (week-based)'
    )
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    print(args)

    dname, writef = process_raw_data(
        args.dataset_name,
        dname2paths,
        args.split_mode,
        args.time_info,
        args.test_ratio,
        args.min_seq_len
    )
    
    split_dataset(
        dname,
        writef,
        args.dataset_name,
        configf,
        args.min_seq_len,
        args.split_mode,
        args.time_info,
        args.test_ratio,
        args.n_folds,
        args.seed
    )
