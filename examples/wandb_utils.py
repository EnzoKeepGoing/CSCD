# pycd/utils/wandb_utils.py
import os
import json
import uuid
import torch
import pandas as pd
from pycd.train.trainer import Trainer
# os.environ['WANDB__SERVICE_WAIT'] = '300'
from pycd.evaluate.metrics import accuracy, rmse, auc

def init_wandb_run(params):
    """
    初始化wandb运行
    - 默认使用wandb自动生成的随机名称（形容词-名词-数字）
    - 如果指定了run_name，则使用指定的名称
    """
    if "use_wandb" not in params or params['use_wandb'] != 1:
        return None
    
    try:
        import wandb
        
        # 项目名称: 优先使用用户指定的，否则自动生成
        project_name = params.get('project_name', None)
        if not project_name:
            model_name = params.get('model_name', 'model')
            project_name = f"pycd-{model_name}"
        
        # 初始化参数
        init_args = {"project": project_name}
        
        # 只有明确提供了run_name时才使用自定义名称
        if 'run_name' in params and params['run_name']:
            init_args["name"] = params['run_name']
        
        # 初始化wandb
        # init_args["mode"] = "offline"
        wandb_run = wandb.init(**init_args)
        
        print(f"Wandb: PROJECT '{project_name}', RUN '{wandb_run.name}'")
        return wandb
        
    except ImportError:
        print("Warning: wandb not installed. Running without wandb tracking.")
        return None



def log_metrics(wandb_instance, metrics):
    """记录指标到wandb（如果启用）"""
    if wandb_instance:
        wandb_instance.log(metrics)

def log_model(wandb_instance, model_path, model=None, aliases=None):
    """将模型保存到wandb（如果启用）"""
    if wandb_instance:
        try:
            import os
            if not os.path.exists(model_path):
                print(f"warning: model file not found: {model_path}")
                # 尝试查找同一目录下的其他模型文件
                model_dir = os.path.dirname(model_path)
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                
                if model_files:
                    alt_path = os.path.join(model_dir, model_files[0])
                    print(f"find alternative model file: {alt_path}")
                    model_path = alt_path
                elif model is not None:
                    # 如果提供了模型对象，尝试重新保存
                    print(f"try to save model to: {model_path}")
                    try:
                        torch.save(model.state_dict(), model_path)
                        if os.path.exists(model_path):
                            print(f"model saved successfully: {model_path}")
                        else:
                            print(f"error: cannot save model: {model_path}")
                            return
                    except Exception as save_e:
                        print(f"error: cannot save model: {save_e}")
                        return
                else:
                    print(f"cannot save model: file not found and no model object provided")
                    return
            
            # 验证文件是否存在且有效
            if os.path.exists(model_path) and os.path.isfile(model_path):
                file_size = os.path.getsize(model_path)
                if file_size == 0:
                    print(f"warning: model file size is 0: {model_path}")
                    return
                    
                print(f"Save model to wandb: {model_path} (size: {file_size} bytes)")
                artifact = wandb_instance.Artifact(
                    name=f"model-{wandb_instance.run.id}", 
                    type="model",
                    description="trained model checkpoint")
                artifact.add_file(model_path)
                wandb_instance.log_artifact(artifact, aliases=aliases)
            else:
                print(f"warning: model file not found or not a valid file: {model_path}")
        except Exception as e:
            print(f"warning: cannot save model to wandb: {e}")
            
def finish_run(wandb_instance):
    """完成wandb运行（如果启用）"""
    if wandb_instance:
        wandb_instance.finish()

def _is_wandb_compatible(value):
    """
    检查值是否与wandb兼容（只允许基本数据类型）
    """
    # 只允许基本的Python数据类型
    if isinstance(value, (str, int, float, bool)):
        return True
    elif isinstance(value, (list, tuple)):
        # 检查列表/元组中的所有元素都是基本类型
        return all(_is_wandb_compatible(item) for item in value)
    elif isinstance(value, dict):
        # 检查字典中的所有键值都是基本类型
        return all(isinstance(k, str) and _is_wandb_compatible(v) for k, v in value.items())
    else:
        # 拒绝所有复杂对象（张量、模型、图结构等）
        return False

def collect_hyperparams(params, model_params):
    """
    只收集基本数据类型的超参数，直接跳过复杂对象
    
    参数:
        params: 完整的参数字典
        model_params: 模型特定的参数字典
        
    返回:
        hyperparams: 只包含基本数据类型的超参数字典
    """
    # 只收集基本信息
    hyperparams = {}
    
    # 添加基本训练信息
    basic_info = ['dataset', 'fold', 'seed', 'model_name']
    for key in basic_info:
        if key in params and _is_wandb_compatible(params[key]):
            hyperparams[key] = params[key]
    
    # 添加训练超参数（只要是基本数据类型）
    train_params = ['lr', 'batch_size', 'epochs', 'hidden_dims1', 'hidden_dims2', 
                   'dropout1', 'dropout2', 'emb_dim', 'latent_dim', 'feature_dim', 
                   'layers', 'n_hid', 'weight_decay', 'alr', 'ratio', 'k', 
                   'lam_seq', 'lam_res']
    
    # 从params中收集基本类型的参数
    for key, value in params.items():
        if key in train_params and _is_wandb_compatible(value):
            hyperparams[key] = value
    
    # 从model_params中收集基本类型的参数
    for key, value in model_params.items():
        if key not in hyperparams and _is_wandb_compatible(value):
            hyperparams[key] = value
    
    return hyperparams


def cleanup_experiment_dir(exp_dir):
    """
    清理实验目录，只保留三个必要文件：config.txt, model.pth, test_predictions.txt
    """
    required_files = {'config.txt', 'model.pth', 'test_predictions.txt'}
    
    for filename in os.listdir(exp_dir):
        if filename not in required_files:
            file_path = os.path.join(exp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
    
    # print(f"Cleanup completed. Experiment directory contains only: {list(required_files)}")


def save_test_predictions(test_loader, model, exp_dir, device='cuda'):
    """
    保存测试集预测结果到 test_predictions.txt
    
    Args:
        test_loader: 测试集数据加载器
        model: 训练好的模型
        exp_dir: 实验目录
        device: 设备
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 根据不同模型类型处理输入
            if len(batch) == 5:  # 带 A：user_id, item_id, q_vector, a_vector, label
                user_ids, item_ids, q_vectors, a_vectors, labels = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                q_vectors = q_vectors.to(device)
                a_vectors = a_vectors.to(device)
                labels = labels.to(device)
                try:
                    outputs = model(user_ids, item_ids, q_vectors, a_vectors)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                except Exception as e:
                    print(f"Model forward error (with A): {e}")
                    continue
            
            elif len(batch) == 4:  # 不带 A：兼容旧模型
                user_ids, item_ids, q_vectors, labels = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                q_vectors = q_vectors.to(device)
                labels = labels.to(device)
                try:
                    outputs = model(user_ids, item_ids, q_vectors)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                except Exception as e:
                    print(f"Model forward error: {e}")
                    continue
            
            else:
                print(f"Unexpected batch format with {len(batch)} elements")
                continue

            
            # 转换为概率和预测标签
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            probs = torch.sigmoid(outputs) if not torch.all((outputs >= 0) & (outputs <= 1)) else outputs
            pred_labels = (probs >= 0.5).float()
            
            # 收集结果
            for i in range(len(user_ids)):
                # 安全地提取张量值，处理0维张量情况
                def safe_item(tensor):
                    if tensor.dim() == 0:  # 0维张量
                        return tensor.item()
                    else:  # 1维或更高维张量
                        return tensor.cpu().item()

                predictions.append({
                    'user_id': int(safe_item(user_ids[i])),
                    'question_id': int(safe_item(item_ids[i])),
                    'correct': float(safe_item(labels[i])),
                    'predict_correct': int(safe_item(pred_labels[i])),
                    'predict_proba': float(safe_item(probs[i]))
                })
    
    # 保存为CSV格式
    predictions_df = pd.DataFrame(predictions)
    predictions_path = os.path.join(exp_dir, 'test_predictions.txt')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")
    
    return predictions_path

class WandbTrainer:
    """
    包装 Trainer 类，只添加 wandb 日志记录功能，
    所有训练逻辑依赖原始 Trainer 实现
    """
    def __init__(self, model, optimizer, scheduler=None, device='cpu', 
             early_stop=None, ckpt_path=None, wandb_instance=None,
             trainer_class=None):
        # 创建一个真正的 Trainer 实例作为内部属性
        trainer_cls = trainer_class or Trainer
        self.trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            early_stop=early_stop,
            ckpt_path=ckpt_path
        )
        self.wandb = wandb_instance

    def __getattr__(self, name):
        """
        代理所有未定义的属性和方法到内部的 trainer 实例
        """
        return getattr(self.trainer, name)

    def fit(self, train_loader, val_loader, metrics_fn, epochs=10, extra_inputs=None,extra_params=None):
        """
        包装原始的 fit 方法，添加 wandb 日志记录
        """
        best_metric = None
        best_epoch = -1
        
        for epoch in range(1, epochs + 1):
            # 使用原始 trainer 训练和评估
            train_loss = self.trainer.train_epoch(train_loader, extra_inputs)
            val_metric = self.trainer.eval_epoch(val_loader, metrics_fn, extra_inputs,extra_params)
                        
            val_acc = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: accuracy(m, t, p, pa), extra_inputs, extra_params)
            val_rmse = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: rmse(m, t, p, pa), extra_inputs, extra_params)
            # 更新学习率调度器
            # if self.trainer.scheduler:
            #     try:
            #         self.trainer.scheduler.step(val_metric) 
            #     except TypeError:
            #         self.trainer.scheduler.step()  
            # 获取当前学习率
            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            
            # 记录到 wandb (不影响原始训练过程)
            if self.wandb:
                self.log_metrics({
                    'epoch': epoch
                })
            
            # 只记录进度，不干预训练逻辑
            improved = (best_metric is None or val_metric > best_metric)
            if improved:
                best_metric = val_metric
                best_epoch = epoch
            
            # 打印当前状态 (使用原始格式)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Metric: {val_metric:.16f}")
            if improved and self.trainer.ckpt_path:
                print(f"  → Saved best model to {self.trainer.ckpt_path}")


            # 添加早停检查
            if self.trainer.early_stop is not None:
                if self.trainer.early_stop.step(val_metric):
                    print(f"  → Early stopping at epoch {epoch}")
                    break
        
        # 最终结果
        return best_metric, best_epoch
    
    def log_metrics(self, metrics):
        """记录指标到 wandb"""
        if self.wandb:
            self.wandb.log(metrics)