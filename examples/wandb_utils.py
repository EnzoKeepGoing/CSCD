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

    if "use_wandb" not in params or params['use_wandb'] != 1:
        return None
    
    try:
        import wandb
        
        project_name = params.get('project_name', None)
        if not project_name:
            model_name = params.get('model_name', 'model')
            project_name = f"pycd-{model_name}"
        

        init_args = {"project": project_name}
        

        if 'run_name' in params and params['run_name']:
            init_args["name"] = params['run_name']
        

        # init_args["mode"] = "offline"
        wandb_run = wandb.init(**init_args)
        
        print(f"Wandb: PROJECT '{project_name}', RUN '{wandb_run.name}'")
        return wandb
        
    except ImportError:
        print("Warning: wandb not installed. Running without wandb tracking.")
        return None



def log_metrics(wandb_instance, metrics):

    if wandb_instance:
        wandb_instance.log(metrics)

def log_model(wandb_instance, model_path, model=None, aliases=None):

    if wandb_instance:
        try:
            import os
            if not os.path.exists(model_path):
                print(f"warning: model file not found: {model_path}")

                model_dir = os.path.dirname(model_path)
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                
                if model_files:
                    alt_path = os.path.join(model_dir, model_files[0])
                    print(f"find alternative model file: {alt_path}")
                    model_path = alt_path
                elif model is not None:

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

    if wandb_instance:
        wandb_instance.finish()

def _is_wandb_compatible(value):


    if isinstance(value, (str, int, float, bool)):
        return True
    elif isinstance(value, (list, tuple)):

        return all(_is_wandb_compatible(item) for item in value)
    elif isinstance(value, dict):

        return all(isinstance(k, str) and _is_wandb_compatible(v) for k, v in value.items())
    else:

        return False

def collect_hyperparams(params, model_params):


    hyperparams = {}
    
    basic_info = ['dataset', 'fold', 'seed', 'model_name']
    for key in basic_info:
        if key in params and _is_wandb_compatible(params[key]):
            hyperparams[key] = params[key]
    
    train_params = ['lr', 'batch_size', 'epochs', 'hidden_dims1', 'hidden_dims2', 
                   'dropout1', 'dropout2', 'emb_dim', 'latent_dim', 'feature_dim', 
                   'layers', 'n_hid', 'weight_decay', 'alr', 'ratio', 'k', 
                   'lam_seq', 'lam_res']
    
    for key, value in params.items():
        if key in train_params and _is_wandb_compatible(value):
            hyperparams[key] = value
    
    for key, value in model_params.items():
        if key not in hyperparams and _is_wandb_compatible(value):
            hyperparams[key] = value
    
    return hyperparams


def cleanup_experiment_dir(exp_dir):
    """
    config.txt, model.pth, test_predictions.txt
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

    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 5:  #  A：user_id, item_id, q_vector, a_vector, label
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
            
            elif len(batch) == 4:  
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

            

            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            probs = torch.sigmoid(outputs) if not torch.all((outputs >= 0) & (outputs <= 1)) else outputs
            pred_labels = (probs >= 0.5).float()
            

            for i in range(len(user_ids)):

                def safe_item(tensor):
                    if tensor.dim() == 0: 
                        return tensor.item()
                    else:  
                        return tensor.cpu().item()

                predictions.append({
                    'user_id': int(safe_item(user_ids[i])),
                    'question_id': int(safe_item(item_ids[i])),
                    'correct': float(safe_item(labels[i])),
                    'predict_correct': int(safe_item(pred_labels[i])),
                    'predict_proba': float(safe_item(probs[i]))
                })
    

    predictions_df = pd.DataFrame(predictions)
    predictions_path = os.path.join(exp_dir, 'test_predictions.txt')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")
    
    return predictions_path

class WandbTrainer:

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

        return getattr(self.trainer, name)

    def fit(self, train_loader, val_loader, metrics_fn, epochs=10, extra_inputs=None,extra_params=None):

        best_metric = None
        best_epoch = -1
        
        for epoch in range(1, epochs + 1):

            train_loss = self.trainer.train_epoch(train_loader, extra_inputs)
            val_metric = self.trainer.eval_epoch(val_loader, metrics_fn, extra_inputs,extra_params)
                        
            val_acc = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: accuracy(m, t, p, pa), extra_inputs, extra_params)
            val_rmse = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: rmse(m, t, p, pa), extra_inputs, extra_params)

            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            
            if self.wandb:
                self.log_metrics({
                    'epoch': epoch
                })
            
            improved = (best_metric is None or val_metric > best_metric)
            if improved:
                best_metric = val_metric
                best_epoch = epoch
            
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Metric: {val_metric:.16f}")
            if improved and self.trainer.ckpt_path:
                print(f"  → Saved best model to {self.trainer.ckpt_path}")


            if self.trainer.early_stop is not None:
                if self.trainer.early_stop.step(val_metric):
                    print(f"  → Early stopping at epoch {epoch}")
                    break
        
        return best_metric, best_epoch
    
    def log_metrics(self, metrics):
        """ wandb"""
        if self.wandb:
            self.wandb.log(metrics)
