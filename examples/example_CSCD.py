import argparse
from wandb_train_test import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KSCD model')
    parser.add_argument('--model_name', type=str, default='kscd')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='peiyou')
    
    # Model hyperparameters
    parser.add_argument('--emb_dim', type=int, default=90)
    parser.add_argument('--lr', type=float, default=0.0018)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb for tracking')
    # Save path
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Save directory, if None, use default path')

    args = parser.parse_args()
    
    params = vars(args)
    params["a_dim"] = 11
    params["a_csv_name"] = "A_matrix.csv"   # 你实际的 A 文件名
    main(params)