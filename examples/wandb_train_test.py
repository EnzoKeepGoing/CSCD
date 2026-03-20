import os
import json
import torch
import pickle
import scipy as sp
import pandas as pd
import numpy as np
from argparse import Namespace
from pycd.preprocess.utils import set_seed
from pycd.utils.logging import init_logger, get_experiment_dir, save_experiment_config
from data.dataset import CDMDataset, RCDDataset, load_question_mapping, load_q_matrix
from pycd.train.trainer import Trainer, EarlyStopping
from pycd.evaluate.metrics import accuracy, auc, rmse,doa
from pycd.evaluate.metrics import accuracy, rmse, auc

from torch.utils.data import DataLoader
from pycd.models.init_model import create_model
from data.utils import batch_convert_to_rcd_json
from wandb_utils import (
    init_wandb_run, log_metrics, 
    log_model, finish_run, collect_hyperparams, WandbTrainer,
    save_test_predictions, cleanup_experiment_dir
)
from data.graph_utils import construct_local_map,construct_kc_kc_graph,construct_kc_ques_graph,process_edge,construct_stu_ques_graph,disengcd_get_file
# from data.utils import load_A_like_Q_matrix 
from torch.utils.data import Dataset
from data.utils import load_A_like_Q_matrix


class InjectADataset(Dataset):
    def __init__(self, base_ds, a_matrix_np):
        self.base = base_ds
        self.A = torch.tensor(a_matrix_np, dtype=torch.float32)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        stu_id, exer_id, q_vector, label = self.base[idx]
        a_vector = self.A[exer_id]
        return stu_id, exer_id, q_vector, a_vector, label
def process_common_args(args_or_dict, model_name=None):
    """
    处理通用参数，填充默认值，可以处理字典或Namespace对象
    
    参数:
        args_or_dict: 命令行参数对象或字典
        model_name: 模型名称，用于设置默认保存目录
        
    返回:
        处理后的参数字典
    """
    if not isinstance(args_or_dict, dict):
        args_dict = vars(args_or_dict)
    else:
        args_dict = args_or_dict

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print('script_dir')
    print(script_dir)
    proj_root = os.path.abspath(os.path.join(script_dir, '..'))
    print('proj_root')
    print(proj_root)
    model = model_name if model_name else args_dict.get('model', 'default')

    if 'data_dir' not in args_dict or args_dict['data_dir'] is None:
        if 'dataset' in args_dict:
            args_dict['data_dir'] = os.path.join(proj_root, 'data', args_dict['dataset'])
            args_dict['base_dir'] = os.path.join(proj_root, 'data')
        else:
            args_dict['data_dir'] = os.path.join(proj_root, 'data', 'default')
            args_dict['base_dir'] = os.path.join(proj_root, 'data', 'default')

    if 'device' not in args_dict or args_dict['device'] is None:
        args_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'save_dir' not in args_dict or args_dict['save_dir'] is None:
        args_dict['save_dir'] = os.path.join(script_dir, 'model_save', model)

    os.makedirs(args_dict['save_dir'], exist_ok=True)

    if 'seed' not in args_dict or args_dict['seed'] is None:
        args_dict['seed'] = 42

    if 'batch_size' not in args_dict:
        args_dict['batch_size'] = 256

    if 'epochs' not in args_dict:
        args_dict['epochs'] = 30

    if 'fold' not in args_dict:
        args_dict['fold'] = 0

    if model_name == 'rcd':
        mapping_path = os.path.join(os.path.dirname(__file__), '..', 'data', args_dict['dataset'], 'id_mapping.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)

        args_dict['exer_n'] = len(id_mapping.get("questions", {}))
        args_dict['knowledge_n'] = len(id_mapping.get("concepts", {}))
        args_dict['student_n'] = len(id_mapping.get("uid", {}))
    return args_dict

def main(params):
    params = process_common_args(params, params.get('model_name', None))

    wandb = None
    if params.get('use_wandb', 0) == 1:
        wandb = init_wandb_run(params)

    args = Namespace(**params)
    set_seed(params['seed'])

    data_dir = params['data_dir']
    base_dir = params['base_dir']
    mapping_path = os.path.join(data_dir, 'id_mapping.json')
    q_matrix_path = os.path.join(data_dir, 'q_matrix.csv')
    train_valid_csv = os.path.join(data_dir, 'train_valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    fold = params['fold']

    question2idx = load_question_mapping(mapping_path)
    q_matrix = load_q_matrix(q_matrix_path, None, num_exercises=len(question2idx), num_concepts=None)
    concept_count = q_matrix.shape[1]

    a_dim = int(params.get("a_dim", 0))
    params["a_dim"] = a_dim  # 确保 create_model 能读到
    
    if a_dim > 0:
        a_csv_path = os.path.join(data_dir, params.get("a_csv_name", "A_matrix.csv"))
        A = load_A_like_Q_matrix(
            a_csv_path=a_csv_path,
            num_exercises=len(question2idx),
            num_concepts=a_dim
        )
    else:
        A = None
    params["q_matrix"]=q_matrix
    params["eval_logs"]=test_csv

    df_train_valid = pd.read_csv(train_valid_csv)
    user_count = int(df_train_valid['user_id'].max()) + 1

    init_logger()
    print(f"Training config: model={params['model_name']}, dataset={params['dataset']}, device={params['device']}, fold={fold}")
    if wandb:
        print(f"Wandb tracking: enabled, project={wandb.run.project}, run={wandb.run.name}")

    if params['model_name'] in ['dina', 'irt', 'mirt', 'neuralcdm', 'kancd', 'kscd']:
        train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
        valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
        test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)
        if params['model_name'] == 'neuralcdm' and a_dim > 0 and A is not None:
            train_ds = InjectADataset(train_ds, A)
            valid_ds = InjectADataset(valid_ds, A)
            test_ds  = InjectADataset(test_ds, A)
        if params['model_name'] == 'kancd' and a_dim > 0 and A is not None:
            train_ds = InjectADataset(train_ds, A)
            valid_ds = InjectADataset(valid_ds, A)
            test_ds  = InjectADataset(test_ds, A)
        if params['model_name'] == 'kscd' and a_dim > 0 and A is not None:
            train_ds = InjectADataset(train_ds, A)
            valid_ds = InjectADataset(valid_ds, A)
            test_ds  = InjectADataset(test_ds, A)
    else:
        if params['model_name'] == 'rcd':
            batch_convert_to_rcd_json(data_dir)
            train_json = os.path.join(data_dir, 'train_valid.json')
            test_json = os.path.join(data_dir, 'test.json')
            train_ds = RCDDataset(train_json)
            valid_ds = RCDDataset(train_json) 
            test_ds = RCDDataset(test_json)

        elif params['model_name'] in ['icdm', 'scd']:

            if params['model_name'] == 'icdm':
                from pycd.models.icdm import build_graph4CE, build_graph4SE, build_graph4SC
                import numpy as np
                np_train =  pd.read_csv(train_valid_csv)
                np_train = np_train[np_train['fold'] != fold].reset_index(drop=True)
                params['np_train'] = np_train.values.astype(np.float64)
                np_val =  pd.read_csv(train_valid_csv)
                np_val = np_val[np_val['fold'] == fold].reset_index(drop=True)
                params['np_val'] = np_val.values.astype(np.float64)
                np_test =  pd.read_csv(test_csv)
                params['np_test'] = np_test.values.astype(np.float64)
                params['q'] =  torch.tensor(q_matrix).to(params['device'])
                params['stu_num'] = user_count
                params['prob_num'] = len(question2idx)
                params['know_num'] = concept_count
                right, wrong = build_graph4SE(params)
                graph_dict = {
                'right': right,
                'wrong': wrong,
                'Q': build_graph4CE(params),
                'I': build_graph4SC(params)
                }
                params['graph_dict'] = graph_dict
                args = Namespace(**params)
                model, model_params, optimizer = create_model(args, concept_count, len(question2idx), user_count)
                
                # 创建实验目录并保存配置
                exp_dir = get_experiment_dir(params['save_dir'], params['model_name'], params['dataset'], params=model_params, seed=params['seed'])
                save_experiment_config(exp_dir, params['model_name'], params['dataset'], model_params)
                
                # 设置模型保存路径
                ckpt_path = os.path.join(exp_dir, 'model.pth')

                #返回的是测试集的结果，验证集只打印不返回
                test_auc, test_accuracy, test_rmse, test_f1, test_doa, _, best_metric, best_epoch = model.train(params['np_train'], params['np_val'], 
                                                                                                                        params['np_test'], q=params['q'], 
                                                                                                                        batch_size=params['batch_size'], 
                                                                                                                        exp_dir=exp_dir, wandb_instance=wandb, 
                                                                                                                        epoch=params['epochs'], lr=params['lr'])
    
                results = standardization(wandb, exp_dir, model_params, params, test_auc, test_accuracy, ckpt_path, best_metric, best_epoch, rmse_=test_rmse, f1_=test_f1, doa_=test_doa)

            elif params['model_name'] == 'scd':
                train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
                valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
                test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)

                train_ds = InjectADataset(train_ds, A)
                valid_ds = InjectADataset(valid_ds, A)
                test_ds  = InjectADataset(test_ds, A)
                params['train_size'] = len(train_ds)
                params['train_loader'] = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
                params['valid_loader'] = DataLoader(valid_ds, batch_size=params['batch_size'])
                params['test_loader'] = DataLoader(test_ds, batch_size=params['batch_size'])
                params['q_matrix'] = q_matrix

                args = Namespace(**params)
                model, model_params, optimizer = create_model(args, concept_count, len(question2idx), user_count)
                
                # 创建实验目录并保存配置
                exp_dir = get_experiment_dir(params['save_dir'], params['model_name'], params['dataset'], params=model_params, seed=params['seed'])
                save_experiment_config(exp_dir, params['model_name'], params['dataset'], model_params)
                
                # 设置模型保存路径
                ckpt_path = os.path.join(exp_dir, 'model.pth')
                
                test_auc, test_accuracy, test_f1, _, best_metric, best_epoch = model.train(epochs=params['interaction_epochs'], nn_epochs=params['parameter_epochs'], 
                                                                                                   dir=exp_dir, lr=params['lr'], population_size=params['population_size'], 
                                                                                                   ngen=params['ngen'], cxpb=params['cxpb'], mutpb=params['mutpb'], wandb_instance=wandb)
                print('test_auc:',test_auc)
                print('test_accuracy:',test_accuracy)
                print('f1_:',f1_)
                results = standardization(wandb, exp_dir, model_params, params, test_auc, test_accuracy, ckpt_path, best_metric, best_epoch, f1_=test_f1)
            
            else:
                results = None

            return results
        
        elif params['model_name'] == 'hypercdm':
            print(" Hypercdm，构建超图结构...")
            from pycd.models.hypercdm import HyperCDM, extract_response_logs, build_r_matrix
            train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
            valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
            test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)

        elif params['model_name'] == 'disengcd':
            from pycd.models.disengcd import sparse_mx_to_torch_sparse_tensor,normalize_row,normalize_sym,crop_tensors_to_smallest_square
            import numpy as np
            params['user_n'] = user_count
            params['exer_n'] = len(question2idx)
            params['knowledge_n'] = concept_count
            print(" Disengcd，构建meta multigraph结构...")
            train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
            valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
            test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)
            data_txt_path = os.path.join(base_dir, params["dataset"], "data.txt")
            if os.path.exists(data_txt_path):
                construct_kc_kc_graph(params["dataset"],base_dir)
                process_edge(params["dataset"],base_dir)
                construct_kc_ques_graph(params["dataset"],base_dir)
                construct_stu_ques_graph(params["dataset"],base_dir)
                disengcd_get_file(params["dataset"],base_dir)
            else:
                print(f"警告: 找不到文件 {data_txt_path}，跳过图构建步骤")

            with open(os.path.join(data_dir,'graph/edges.pkl'), "rb") as f: 
                edges = pickle.load(f) 
                f.close()
            edges2=edges
            adjs_pt = []
            for mx in edges:
                mx_tensor=mx.astype(np.float32).toarray()
                id_matrix=sp.eye(mx.shape[0], dtype=np.float32)
                mx_tesnor_reshape,id_matrix_reshape=crop_tensors_to_smallest_square(mx_tensor,id_matrix)

                adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                    normalize_row( mx_tesnor_reshape+ id_matrix_reshape)).to(params['device']))
                adjs_pt.append(
                    sparse_mx_to_torch_sparse_tensor(sp.sparse.coo_matrix(sp.eye(params["knowledge_n"]+params["exer_n"]+params["user_n"], dtype=np.float32)).tocoo()).to(params['device'])) 
                adjs_pt.append(torch.sparse.FloatTensor(size=edges[0].shape).to(params['device']))

            node_types = np.zeros((params["knowledge_n"]+params["exer_n"]+params["user_n"],), dtype=np.int32)
            print(node_types.shape)
            a = np.arange(params["knowledge_n"])                                                      
            b = np.arange(params["knowledge_n"],params["knowledge_n"]+params["exer_n"])                                                   
            c = np.arange(params["knowledge_n"]+params["exer_n"],params["user_n"] + params["exer_n"]+params["user_n"])
            node_types[a.shape[0]:a.shape[0] + b.shape[0]] = 1                       
            node_types[a.shape[0] + b.shape[0]:] = 2 
            params["node_types"]=node_types
            params["local_map"]=construct_local_map(params['dataset'],base_dir)
            params["all_map"]=adjs_pt
            args=Namespace(**params)

        elif params['model_name'] == 'orcdf':
            train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
            valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
            test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)

        else:
            print(f"警告: 未实现模型 {params['model_name']} 的数据集处理，使用默认CDMDataset")
            train_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='train', fold=fold)
            valid_ds = CDMDataset(train_valid_csv, question2idx, q_matrix, fold_mode='valid', fold=fold)
            test_ds = CDMDataset(test_csv, question2idx, q_matrix, is_test=True)

    # train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    train_loader = DataLoader(
    train_ds, 
    batch_size=params['batch_size'], 
    shuffle=True,
    num_workers=18,           # 建议：CPU 核心数的一半
    pin_memory=True,
    persistent_workers=True, # Windows 则设 False
    prefetch_factor=4
)

    # valid_loader = DataLoader(valid_ds, batch_size=params['batch_size'])
    # test_loader = DataLoader(test_ds, batch_size=params['batch_size'])


    valid_loader = DataLoader(
        valid_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=18,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=18,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    model, model_params, optimizer = create_model(
        args, 
        concept_count, 
        len(question2idx), 
        user_count
    )

    #hypercdm build graph
    if params['model_name'] == 'hypercdm':
        response_logs = extract_response_logs(train_ds)
        r_matrix = build_r_matrix(train_ds, user_count, len(question2idx))
        model.hyper_build(response_logs, q_matrix, r_matrix)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # ORCDF  build graph
    if params["model_name"] == "orcdf":
        from pycd.models.orcdf import  extract_response_array
        print("ORCDF Build Graph...")
        response_array = extract_response_array(train_ds)
        se_graph_right, se_graph_wrong = [model.extractor._create_adj_se(response_array, is_subgraph=True)[i] for i in range(2)]
        se_graph = model.extractor._create_adj_se(response_array, is_subgraph=False)

        graph_dict = {
            'right': model.extractor._final_graph(se_graph_right, q_matrix),
            'wrong': model.extractor._final_graph(se_graph_wrong, q_matrix),
            'response': response_array,
            'Q_Matrix': q_matrix.copy(),
            'flip_ratio': model.flip_ratio,
            'all': model.extractor._final_graph(se_graph, q_matrix)
        }
        model.extractor.get_graph_dict(graph_dict)
        model.extractor.get_flip_graph()
    
    if wandb:
        from wandb_utils import collect_hyperparams
        hyperparams = collect_hyperparams(params, model_params)
        wandb.config.update(hyperparams)
        print(f"record hyperparams to wandb: {', '.join(hyperparams.keys())}")

    if params['model_name'] == 'disengcd':
        # DisenGCD模型不使用scheduler
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5)

    early_stop = EarlyStopping(patience=5, mode='max')

    model_name = params['model_name']
    dataset_name = params['dataset']
    save_dir = params['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # 创建实验目录并保存配置
    exp_dir = get_experiment_dir(
        save_dir,
        model_name,
        dataset_name,
        params=model_params,
        seed=params['seed']
    )
    save_experiment_config(exp_dir, model_name, dataset_name, model_params)
    
    # 设置模型保存路径
    ckpt_path = os.path.join(exp_dir, 'model.pth')
    print(f"Model save to: {ckpt_path}")
    
    # 判断是否为DisenGCD模型
    if params['model_name'] == 'disengcd':
        # 使用专门为DisenGCD设计的训练器
        from pycd.train.trainer import Trainer4DisenGCD
        trainer = WandbTrainer(
            model=model,
            optimizer=optimizer,  # 这里传入的是优化器列表
            scheduler=None,       # DisenGCD模型不使用scheduler
            device=params['device'],
            early_stop=early_stop,
            ckpt_path=ckpt_path,
            wandb_instance=wandb,
            trainer_class=Trainer4DisenGCD  # 使用专门的训练器类
        )
    else:
        # 其他模型使用普通的Trainer
        trainer = WandbTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=params['device'],
            early_stop=early_stop,
            ckpt_path=ckpt_path,
            wandb_instance=wandb
        )

    # print("====== Eval before training ======")
    # # 👇 打印 ACC 和 RMSE
    # trues, preds = [], []
    # for batch in test_loader:
    #     batch = [x.to(trainer.device) for x in batch]
    #     pred = trainer._call_model(batch)
    #     if isinstance(pred, tuple):
    #         pred = pred[0]
    #     preds.extend(pred.detach().cpu().tolist())
    #     trues.extend(batch[-1].detach().cpu().tolist())
    
    # print("ACC:", accuracy(trainer.model, trues, preds, params))
    # print("RMSE:", rmse(trainer.model, trues, preds, params))

    best_valid_auc, best_epoch = trainer.fit(
        train_loader,
        test_loader,
        #        test_loader,
        metrics_fn=lambda m,t, p,pa: auc(m,t, p,pa),
        epochs=params['epochs'],
        extra_params=params,
    )
    best_valid_auc, best_epoch = trainer.fit(
        train_loader,
        test_loader,
        metrics_fn=lambda m,t,p,pa: auc(m,t,p,pa),
        epochs=params['epochs'],
        extra_params=params,
    )



    
    # best_valid_auc, best_epoch = trainer.fit(
    #     train_loader,
    #     test_loader,
    #     metrics_fn=lambda m, t, p, pa: {
    #         "AUC": auc(m, t, p, pa),
    #         "ACC": accuracy(m, t, p, pa),
    #         "RMSE": rmse(m, t, p, pa)
    #     },
    #     epochs=params['epochs'],
    #     extra_params=params,
    # )


    # 确保模型文件被保存
    if not os.path.exists(ckpt_path):
        # print(f"Warning: Model file not found at {ckpt_path}, saving current model...")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to: {ckpt_path}")

    print(f"Evaluating fold {fold} on test set...")
    test_auc = trainer.eval_epoch(test_loader, lambda m,t, p,pa: auc(m,t, p,pa))
    test_acc = trainer.eval_epoch(test_loader, lambda m,t, p,pa: accuracy(m,t, p,pa))
    test_rmse = trainer.eval_epoch(test_loader, lambda m,t, p,pa: rmse(m,t, p,pa))
    test_doa = trainer.eval_epoch(test_loader, lambda m,t, p,pa: doa(m,t, p,pa),extra_params=params)
    # test_doa = 0

    print(f"Fold {fold} Test AUC: {test_auc:.4f}, Test Accuracy: {test_acc:.4f}, Test RMSE: {test_rmse:.4f}, Test DOA: {test_doa:.4f}")

    # 保存测试集预测结果
    save_test_predictions(test_loader, model, exp_dir, params['device'])

    final_results = {
        'fold': fold,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'test_rmse': test_rmse,
        'best_valid_auc': best_valid_auc,
        'best_epoch': best_epoch,
        'model_params': model_params
    }

    if wandb:
        # 训练结束后，确保模型文件仍然存在
        if not os.path.exists(ckpt_path):
            # 重新保存模型
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to: {ckpt_path}")
            
        if os.path.exists(ckpt_path):
            file_size = os.path.getsize(ckpt_path)
            print(f"model file size: {file_size} bytes, save success")
        else:
            print(f"error: cannot save model file")
                
        # 然后上传
        log_metrics(wandb, {
            'test_auc': test_auc,
            'test_acc': test_acc,
            'test_rmse': test_rmse,
            'ckpt_path': ckpt_path
        })
        # log_model(wandb, ckpt_path, aliases=["best"])

    # 清理实验目录，只保留必要文件
    cleanup_experiment_dir(exp_dir)

    if wandb:
        finish_run(wandb)

    return final_results

def standardization(wandb, exp_dir, model_params, params, auc_, accuracy_, ckpt_path, best_metric, best_epoch, rmse_=None, f1_=None, doa_=None):
    # 对train和eval难以实现的，在model.py实现训练流程，返回结果，通过这个函数格式化对齐
    # 加入wandb
    fold = params['fold']
    if wandb:
        from wandb_utils import collect_hyperparams
        hyperparams = collect_hyperparams(params, model_params)
        wandb.config.update(hyperparams)
        print(f"记录超参数到wandb: {', '.join(hyperparams.keys())}")

    # 测试集评估
    print(f"Evaluating fold {fold} on test set...")

    if params['model_name'] =='scd':
        print(f"Fold {fold} Test AUC: {auc_:.4f}, Test Accuracy: {accuracy_:.4f}, Test F1: {f1_:.4f}")

    if params['model_name'] =='icdm':
        print(f"Fold {fold} Test AUC: {auc_:.4f}, Test Accuracy: {accuracy_:.4f}, Test RMSE: {rmse_:.4f}, Test F1: {f1_:.4f}, Test Doa: {doa_:.4f}")
    
    # 保存实验结果
    results = {
        'fold': fold,
        'test_auc': auc_,
        'test_acc': accuracy_,
        'test_rmse': rmse_,
        'test_f1': f1_,
        'test_doa': doa_,
        'best_valid_auc': best_metric,
        'best_epoch': best_epoch,
        'model_params': model_params
    }

    if wandb:
        # 训练结束后，确保模型文件仍然存在
        if not os.path.exists(ckpt_path):
            print(f"error: cannot save model file") #保存失败
            return None
        else:
            file_size = os.path.getsize(ckpt_path)
            print(f"model file size: {file_size} bytes, save success")
            
        log_metrics(wandb, {
            'test_auc': auc_,
            'test_acc': accuracy_,
            'test_rmse': rmse_,
            'test_f1': f1_,
            'test_doa': doa_,
            'ckpt_path': ckpt_path
        })
        # log_model(wandb, ckpt_path, aliases=["best"]) 不需要上传模型文件

    # 清理实验目录，只保留必要文件
    cleanup_experiment_dir(exp_dir)
    
    if wandb:
        finish_run(wandb)

    return results