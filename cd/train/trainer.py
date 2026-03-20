# PYCD/pycd/train/trainer.py
import torch
from tqdm import tqdm

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=5, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric):
        """
        Returns True if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        improve = False
        if self.mode == 'max' and metric > self.best + self.min_delta:
            improve = True
        elif self.mode == 'min' and metric < self.best - self.min_delta:
            improve = True

        if improve:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

class Trainer:
    """
    通用训练器：负责训练、验证/测试循环；支持早停、学习率调度与模型 checkpoint。
    兼容两种 batch 结构：
      - (u, q, qv, y)           # 无A矩阵
      - (u, q, qv, av, y)       # 有A矩阵（方式A）
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device='cpu',
        early_stop: EarlyStopping = None,
        ckpt_path: str = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stop = early_stop
        self.ckpt_path = ckpt_path

    # ---- 统一的安全调用：自动适配有/无 A 的模型 & 批次 ----
    def _call_model(self, batch, extra_inputs=None):
        args = list(batch[:-1])  # 去掉 label
        if extra_inputs:
            args += list(extra_inputs)
        try:
            return self.model(*args)  # 新模型 forward(u,q,qv[,av])
        except TypeError:
            # 老模型只有 (u,q,qv)
            if len(args) >= 4:
                return self.model(*args[:3])
            raise

    def train_epoch(self, dataloader, extra_inputs=None):
        if hasattr(self.model, 'on_epoch_start'):  # orcdf
            self.model.on_epoch_start()

        self.model.train()
        losses = []
        for batch in tqdm(dataloader, desc='Train'):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad(set_to_none=True)
            preds = self._call_model(batch, extra_inputs)
            loss = self.model.loss(preds, batch[-1])
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model, 'apply_clipper'):  # rcd & orcdf
                self.model.apply_clipper()
            losses.append(loss.item())

        if self.scheduler:
            try:
                self.scheduler.step(sum(losses) / len(losses))
            except TypeError:
                self.scheduler.step()
        return sum(losses) / len(losses)

    def eval_epoch(self, dataloader, metrics_fn, extra_inputs=None, extra_params=None):
        """
        在验证/测试集上评估，返回 metrics_fn 计算的结果。
        metrics_fn(trues, preds) 应返回单个指标或字典。
        """
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval'):
                batch = [x.to(self.device) for x in batch]
                pred = self._call_model(batch, extra_inputs)
                if isinstance(pred, tuple):  # orcdf forward可能返回 (logits, extra_loss)
                    pred = pred[0]
                preds.extend(pred.detach().cpu().tolist())
                trues.extend(batch[-1].detach().cpu().tolist())
        return metrics_fn(self.model, trues, preds, extra_params)

    # def fit(
    #     self,
    #     train_loader,
    #     val_loader,
    #     metrics_fn,
    #     epochs: int = 10,
    #     extra_inputs=None,
    #     extra_params=None,
    # ):
    #     """
    #     完整的训练-验证流程，支持早停、学习率调度与模型 checkpoint。
    #     """
    #     best_metric = None
    #     for epoch in range(1, epochs + 1):
    #         train_loss = self.train_epoch(train_loader, extra_inputs)
    #         print('进来')
    #         val_metric = self.eval_epoch(val_loader, metrics_fn, extra_inputs, extra_params)
    #         val_acc = self.eval_epoch(val_loader, lambda m, t, p, pa: accuracy(m, t, p, pa), extra_inputs, extra_params)
    #         val_rmse = self.eval_epoch(val_loader, lambda m, t, p, pa: rmse(m, t, p, pa), extra_inputs, extra_params)
    #         print(f"        ACC: {val_acc:.4f}, RMSE: {val_rmse:.4f}")

    #         print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Metric: {val_metric}")

    #         # 保存最优模型
    #         if self.ckpt_path is not None:
    #             if best_metric is None or val_metric > best_metric:
    #                 best_metric = val_metric
    #                 torch.save(self.model.state_dict(), self.ckpt_path)
    #                 print(f"  → Saved best model to {self.ckpt_path}")

    #         # 早停判断
    #         if self.early_stop is not None and self.early_stop.step(val_metric):
    #             print(f"  → Early stopping at epoch {epoch}")
    #             break

    #     # 训练结束后，加载最优参数
    #     if self.ckpt_path is not None:
    #         self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
    def fit(self, train_loader, val_loader, metrics_fn, epochs=10, extra_inputs=None, extra_params=None):
        best_metric = None
        best_epoch = -1
        
        for epoch in range(1, epochs + 1):
            train_loss = self.trainer.train_epoch(train_loader, extra_inputs)
            val_metric = self.trainer.eval_epoch(val_loader, metrics_fn, extra_inputs, extra_params)
            val_acc = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: accuracy(m, t, p, pa), extra_inputs, extra_params)
            val_rmse = self.trainer.eval_epoch(val_loader, lambda m, t, p, pa: rmse(m, t, p, pa), extra_inputs, extra_params)
    
            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            
            # logging to wandb
            if self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_auc': val_metric,
                    'val_acc': val_acc,
                    'val_rmse': val_rmse,
                    'lr': current_lr
                })
            
            improved = (best_metric is None or val_metric > best_metric)
            if improved:
                best_metric = val_metric
                best_epoch = epoch
            
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, AUC: {val_metric:.4f}, ACC: {val_acc:.4f}, RMSE: {val_rmse:.4f}")
            if improved and self.trainer.ckpt_path:
                print(f"  → Saved best model to {self.trainer.ckpt_path}")
            
            if self.trainer.early_stop is not None:
                if self.trainer.early_stop.step(val_metric):
                    print(f"  → Early stopping at epoch {epoch}")
                    break
        
        return best_metric, best_epoch


class Trainer4DisenGCD(Trainer):
    """
    特化版训练器（保持原有逻辑），同样通过 _call_model 兼容有/无 A。
    """
    def __init__(self, model, optimizer, scheduler=None, device='cpu',
                 early_stop: EarlyStopping = None, ckpt_path: str = None):
        super().__init__(model, optimizer, scheduler, device, early_stop, ckpt_path)
        self.model = model
        self.optimizer = optimizer[0]
        self.optimizer2 = optimizer[1]
        self.scheduler = scheduler
        self.device = device
        self.early_stop = early_stop
        self.ckpt_path = ckpt_path

    def train_epoch(self, dataloader, extra_inputs=None):
        self.model.train()
        losses = []
        progress_bar = tqdm(dataloader, desc='Train')
        for batch in progress_bar:
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad(set_to_none=True)
            output_1 = self._call_model(batch, extra_inputs)  # logits in [0,1]
            output_0 = torch.ones_like(output_1) - output_1
            output = torch.cat((output_0, output_1), dim=1)  # [B,2]
            loss_w = self.model.loss(torch.log(output + 1e-10), batch[-1])
            loss_w.backward()
            self.optimizer.step()
            losses.append(loss_w.item())
            progress_bar.set_description(f"Train (loss={loss_w.item():.4f})")
        return sum(losses) / len(losses) if losses else 0.0

    def eval_epoch(self, dataloader, metrics_fn, extra_inputs=None, extra_params=None):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval'):
                batch = [x.to(self.device) for x in batch]
                output_1 = self._call_model(batch, extra_inputs)
                preds.extend(output_1.view(-1).detach().cpu().tolist())
                trues.extend(batch[-1].detach().cpu().tolist())
        return metrics_fn(self.model, trues, preds, extra_params)
