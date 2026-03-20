# pycd/evaluate/__init__.py

from .metrics import accuracy, auc, rmse, doa  # 如有其他指标，也一并导入

__all__ = [
    "accuracy",
    "auc",
    "rmse",
    "doa",
    # 如果后续增加其他评价方法，也写在这里
]