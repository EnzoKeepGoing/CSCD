#!/usr/bin/env python
import os, sys
import json
import argparse

def main(args):
    log_file = args.log_file
    output_script = args.output_script
    start_sweep = args.start_sweep
    end_sweep = args.end_sweep
    dataset_name = args.dataset
    model_name = args.model_name
    gpu_ids = args.gpu_ids.split(",")
    project_name = args.project_name if args.project_name else "pycd-experiments"
    
    # 加载wandb配置
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if not WANDB_API_KEY:
        try:
            with open("../configs/wandb.json") as fin:
                wandb_config = json.load(fin)
                WANDB_API_KEY = wandb_config["api_key"]
        except Exception as e:
            print(f"Error loading wandb config: {e}")
            WANDB_API_KEY = "YOUR_API_KEY"  # 使用你的API密钥作为默认值
    
    print(f"Using WANDB_API_KEY: {WANDB_API_KEY}")
    
    # 读取日志文件解析sweep ID
    sweep_info = []
    try:
        with open(log_file, "r") as fin:
            lines = fin.readlines()
            for i, line in enumerate(lines):
                if "wandb: Creating sweep from:" in line:
                    # 提取文件路径
                    sweep_file = line.strip().split(":")[-1].strip()
                    # 获取文件名，不含路径
                    sweep_file_name = os.path.basename(sweep_file).split(".")[0]
                    
                    # 查找对应的sweep ID
                    for j in range(i, min(i+5, len(lines))):
                        if "wandb: Run sweep agent with:" in lines[j]:
                            sweep_id = lines[j].strip().split(":")[-1].strip()
                            sweep_info.append((sweep_file_name, sweep_id))
                            break
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    print(f"Found {len(sweep_info)} sweeps in log file")
    
    # 过滤并生成命令
    filtered_sweeps = []
    for idx, (file_name, sweep_id) in enumerate(sweep_info):
        if dataset_name in file_name and model_name in file_name:
            filtered_sweeps.append((idx, file_name, sweep_id))
    
    print(f"Filtered to {len(filtered_sweeps)} sweeps matching dataset={dataset_name} and model={model_name}")
    
    # 写入脚本
    with open(output_script, "w") as fout:
        fout.write("#!/bin/bash\n")
        for i, (idx, file_name, sweep_id) in enumerate(filtered_sweeps):
            if i >= start_sweep and i < end_sweep:
                gpu_id = gpu_ids[i % len(gpu_ids)]
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} WANDB_API_KEY={WANDB_API_KEY} nohup wandb agent {sweep_id} > {file_name}_agent.log 2>&1 &\n"
                fout.write(cmd)
                print(f"Added sweep command: {cmd.strip()}")
    
    # 添加执行权限
    os.chmod(output_script, 0o755)
    print(f"Generated sweep agent script: {output_script}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate script to run wandb sweep agents")
    parser.add_argument("log_file", help="Log file containing sweep IDs")
    parser.add_argument("output_script", help="Output script filename")
    parser.add_argument("start_sweep", type=int, help="Start sweep index")
    parser.add_argument("end_sweep", type=int, help="End sweep index")
    parser.add_argument("dataset", help="Dataset name to filter")  # Changed from dataset_name to dataset
    parser.add_argument("model_name", help="Model name to filter")
    parser.add_argument("gpu_ids", help="Comma-separated GPU IDs")
    parser.add_argument("--project_name", help="Wandb project name")
    
    args = parser.parse_args()
    main(args)