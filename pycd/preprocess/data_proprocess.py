import os, sys, json
import pandas as pd

def load_q2c(qname):
    df = pd.read_csv(qname, encoding = "utf-8",low_memory=False).dropna(subset=["name", "topic"])
    dq2c = dict()
    for name, topic in zip(df["name"], df["topic"]):
        if name not in dq2c:
            dq2c[name] = topic
        else:
            print(f"already has topic in dict: {name}: {topic}, {dq2c[name]}")
    print(f"dq2c: {len(dq2c)}")
    return dq2c

def load_q2c_py(fname):
    dq2c = dict()
    with open(fname, "r") as fin:
        obj = json.load(fin)
        for qid in obj:
            cur = obj[qid]
            content = cur["content"]
            concept_routes = cur["concept_routes"]
            analysis = cur["analysis"]
            cs = []
            for route in concept_routes:
                tailc = route.split("----")[-1]
                if tailc not in cs:
                    cs.append(tailc)
            dq2c[qid] = "_".join(cs)
    return dq2c


def process_raw_data(dataset_name, dname2paths, split_mode=1, time_info=0, test_ratio=0.2, min_seq_len=15):
    """
    Process raw data according to dataset name

    Args:
        dataset_name: Name of the dataset
        dname2paths: Dictionary mapping dataset names to paths
        split_mode: Processing strategy mode (1, 2, 3)

    Returns:
        tuple: (data directory, intermediate TXT file path)
    """
    print(f"Start processing dataset: {dataset_name}")
    
    if dataset_name not in dname2paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Dataset path: {readf}")
    print(f"Saved TXT file path: {writef}")

    # Call different processing functions according to different datasets
    # Special handling for math1 and math2
    if dataset_name in ["math1", "math2"] and split_mode != 1:
        raise ValueError(f"Unsupported processing mode, all student problem answer records only once: {split_mode}")
        
    # Call different processing functions according to different datasets
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode)  # Pass split_mode as parameter
        
    elif dataset_name == "math1":
        from .math1_preprocess import read_data_from_txt
        read_data_from_txt(readf, writef)
        
    elif dataset_name == "math2":
        from .math2_preprocess import read_data_from_txt
        read_data_from_txt(readf, writef)
    
    elif dataset_name == "frcsub":
        from .frcsub_preprocess import read_data_from_txt
        read_data_from_txt(readf, writef)

    elif dataset_name == "assist2012":
        from .assist2012_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info)  

    elif dataset_name == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, test_ratio, min_seq_len) 
    
    elif dataset_name == "junyi":
        dq2c = load_q2c(readf.replace("junyi_ProblemLog_original.csv","junyi_Exercise_table.csv"))
        from .junyi_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, dq2c, test_ratio, min_seq_len)  

    elif dataset_name == "ednet":
        from .ednet_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, test_ratio, min_seq_len) 
    
    elif dataset_name == "nips_task34":
        metap = os.path.join(dname, "metadata")
        from .nips_task34_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, metap, "task_3_4", test_ratio, min_seq_len)
    
    elif dataset_name == "slp_math":
        from .slp_math_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info) 

    elif dataset_name == "peiyou":
        fname = readf.split("/")[-1]
        dq2c = load_q2c_py(readf.replace(fname,"questions.json"))
        from .peiyou_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, test_ratio, min_seq_len, dq2c) 
    
    elif dataset_name == "jiuzhang_g3":
        from .jiuzhang_g3_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, test_ratio, min_seq_len)

    elif dataset_name == "jiuzhang_g4_g5" or dataset_name == "jiuzhang_g7":
        from .jiuzhang_g4_g5_g7_preprocess import read_data_from_csv
        read_data_from_csv(readf, writef, split_mode, time_info, test_ratio, min_seq_len)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dname, writef