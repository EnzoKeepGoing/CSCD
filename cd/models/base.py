import torch
import torch.nn as nn

class BaseCDModel(nn.Module):
    """
    抽象基类，定义所有认知诊断模型应实现的接口。
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_ids: torch.LongTensor,
                exercise_ids: torch.LongTensor,
                exercise_features=None) -> torch.Tensor:
        """
        前向计算，返回预测概率张量。
        """
        raise NotImplementedError

    def loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        """
        return nn.BCELoss()(pred, label)

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        推断接口，可直接调用 forward。
        """
        self.eval()
        with torch.no_grad():
            return self.forward(*args, **kwargs)
    
    def get_all_knowledge_emb(self):
        """
        默认返回None，表示模型不支持DOA计算
        """
        return None