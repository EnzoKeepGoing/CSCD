import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCDModel

# ==========================================
# 1. KAN 投影算子 (High-Order Projector)
# ==========================================
class KANProjector(nn.Module):
    """
    KAN 投影头：用于将输入特征映射到高维交互空间。
    Story: 利用正弦/余弦基函数扩展特征流形，增强非线性表达能力。
    """
    def __init__(self, in_dim, out_dim, num_basis=3):
        super().__init__()
        # 基础线性路径
        self.base_linear = nn.Linear(in_dim, out_dim)
        # 谱展开系数
        self.basis_weights = nn.Parameter(torch.Tensor(out_dim, num_basis))
        # 频率控制参数
        self.frequency1 = nn.Linear(in_dim, 1)
        self.frequency2 = nn.Linear(in_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)
        nn.init.normal_(self.basis_weights, std=0.02) # 小方差初始化基函数权重

    def forward(self, x):
        base = self.base_linear(x)
        x_norm = torch.tanh(x) # 归一化防止震荡
        
        # KAN Basis Expansion (SiLU, Sin, Cos)
        basis_out = (
            self.basis_weights[:, 0] * F.silu(base) +
            self.basis_weights[:, 1] * torch.sin(base * torch.pi * self.frequency1(x)) +
            self.basis_weights[:, 2] * torch.cos(base * torch.pi * self.frequency2(x))
        )
        # 残差连接
        return base + basis_out

# ==========================================
# 2. 核心模块: MHB-KAN
# ==========================================
class GradientDetachedContext(nn.Module):
    """
    [Top-Tier Complex Version]
    Name: Multi-Head Bilinear KAN Interaction Network (MHB-KAN)
    
    Paper Story Highlights:
    1. Tensor Fusion: 使用双线性池化捕捉 语义(A) 与 知识(K) 的二阶交互。
    2. Structural Constraint: Q矩阵作为"交互掩码"介入张量运算，防止语义噪声。
    3. Multi-Head: 在多个子空间并行建模，提升鲁棒性。
    4. Near-Zero Init: 使用极小随机噪声初始化，实现热启动。
    """
    def __init__(self, emb_dim, n_concepts, a_dim=11, hidden_dim=64, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 维度检查
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # === A. 双流 KAN 投影器 ===
        # Stream 1: 语义流 (Semantic) - 处理 E+A
        self.semantic_proj = KANProjector(emb_dim + a_dim, hidden_dim)
        
        # Stream 2: 知识流 (Knowledge) - 处理 K
        self.knowledge_proj = KANProjector(emb_dim, hidden_dim)
        
        # === B. 双线性交互核 (Bilinear Core) ===
        self.bilinear_dropout = nn.Dropout(0.01)
        
        # === C. 结构感知融合 (Structure Fusion) ===
        # 将 Q 矩阵 (0/1) 映射为高维特征
        self.q_proj = nn.Linear(1, hidden_dim)
        
        # === D. 输出生成器 ===
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), # KAN 风格激活
            nn.Linear(hidden_dim, emb_dim)
        )
        
        self._init_params()

    def _init_params(self):
        # 默认 Xavier 初始化
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # === [核心] Near-Zero Initialization ===
        # 不使用全0，而是使用极小高斯噪声 (std=1e-4)
        # 目的：保证 Step 0 不影响主网络，但保留微弱梯度流，加速收敛
        nn.init.normal_(self.out_proj[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, k_static, exer_emb, a_vector, q_vector):
        # k_static: [Batch, N, emb]
        # q_vector: [Batch, N]
        
        batch_size = k_static.size(0)
        n_concepts = k_static.size(1)
        
        # 1. Gradient Detach (保护主网络)
        exer_context = exer_emb.detach()
        raw_semantic = torch.cat([exer_context, a_vector], dim=-1)
        
        # ==========================================
        # Step 1: KAN 空间投影 (Multi-Head Projection)
        # ==========================================
        
        # Semantic Query: [Batch, hidden] -> [Batch, 1, hidden]
        s_vec = self.semantic_proj(raw_semantic).unsqueeze(1)
        
        # Knowledge Key: [Batch, N, hidden]
        k_vec = self.knowledge_proj(k_static)
        
        # ==========================================
        # Step 2: 多头拆分 (Multi-Head Splitting)
        # ==========================================
        # Reshape to [Batch, *, Num_Heads, Head_Dim]
        s_heads = s_vec.view(batch_size, 1, self.num_heads, self.head_dim)
        k_heads = k_vec.view(batch_size, n_concepts, self.num_heads, self.head_dim)
        
        # ==========================================
        # Step 3: 双线性交互 (Bilinear Interaction)
        # ==========================================
        # Hadamard Product in subspace
        # [Batch, N, Heads, Dim]
        interaction = s_heads * k_heads
        
        # ==========================================
        # Step 4: 结构约束注入 (Structural Injection)
        # ==========================================
        # q_vector: [Batch, N] -> [Batch, N, 1, 1]
        q_mask = q_vector.view(batch_size, n_concepts, 1, 1)
        
        # Q 特征映射
        q_feat = self.q_proj(q_vector.unsqueeze(-1))
        q_heads = q_feat.view(batch_size, n_concepts, self.num_heads, self.head_dim)
        
        # 融合公式: (交互特征 + 结构特征) * 结构掩码
        # 含义: 只有 Q=1 的位置允许发生强交互，Q=0 的位置被抑制
        fused_heads = (interaction + q_heads) * q_mask
        
        # ==========================================
        # Step 5: 聚合与输出
        # ==========================================
        
        # 合并多头
        fused_feat = fused_heads.view(batch_size, n_concepts, self.hidden_dim)
        fused_feat = self.bilinear_dropout(fused_feat)
        
        # 生成 Delta K
        delta_k = self.out_proj(fused_feat)
        
        return delta_k

# ==========================================
# 3. 主模型: KSCD
# ==========================================
class KSCD(BaseCDModel):
    def __init__(self, n_students, n_exercises, n_concepts, emb_dim=20, a_dim=11):
        super(KSCD, self).__init__()
        self.n_students = n_students
        self.n_exercises = n_exercises
        self.n_concepts = n_concepts
        self.emb_dim = emb_dim
        
        # === KSCD 主路 Embedding ===
        self.student_emb = nn.Embedding(self.n_students, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.n_exercises, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.n_concepts, self.emb_dim))
        
        self.emb_dropout = nn.Dropout(0.2)
        
        nn.init.kaiming_normal_(self.knowledge_emb)

        # === 传统的 CDM 预测层 ===
        self.f_sk = nn.Linear(self.emb_dim * 2, self.n_concepts)
        self.f_ek = nn.Linear(self.emb_dim * 2, self.n_concepts)
        self.f_se = nn.Linear(self.n_concepts, 1)
        
        # === 动态修正模块 (MHB-KAN) ===
        # hidden_dim=64, num_heads=4 是一个比较稳的配置
        self.context_module = GradientDetachedContext(
            emb_dim=emb_dim, n_concepts=n_concepts, a_dim=a_dim, hidden_dim=64, num_heads=4
        )
        
        # Alpha 控制
        # 建议初始值设小一点(0.1)，配合 Near-Zero Init 实现平滑热启动
        self.fusion_alpha = nn.Parameter(torch.tensor(0.1))

        self._init_params()

    def _init_params(self):
        for name, param in self.named_parameters():
            if param.dim() < 2: continue
            if 'weight' in name: nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)
        # context_module 的初始化已经在其内部 _init_params 完成了

    def forward(self, stu_id, exer_id, q_vector, a_vector=None):
        # [Fix] 兼容性处理
        # 确保 trainer 传丢参数时能报错提示，而不是默默失败
        if a_vector is None:
             raise ValueError("a_vector is missing! Please check Trainer inputs.")

        # 1. 准备 Embedding
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(exer_id)
        
        stu_emb_main = self.emb_dropout(stu_emb)
        exer_emb_main = self.emb_dropout(exer_emb)
        
        batch_size = stu_id.size(0)
        
        # 构造静态 K
        k_static = self.knowledge_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # ==========================================
        # 2. 计算动态 K (MHB-KAN)
        # ==========================================
        
        # 传入所有必要的特征: Static K, Exercise, A(Semantic), Q(Structure)
        delta_k = self.context_module(k_static, exer_emb_main, a_vector, q_vector)
        
        # Alpha 缩放
        alpha = F.softplus(self.fusion_alpha)
        
        # K_dynamic = K + alpha * Delta
        k_dynamic = k_static + alpha * delta_k
        
        # ==========================================
        # 3. KSCD 主逻辑
        # ==========================================
        
        stu_emb_exp = stu_emb_main.unsqueeze(1).repeat(1, self.n_concepts, 1)
        exer_emb_exp = exer_emb_main.unsqueeze(1).repeat(1, self.n_concepts, 1)
        
        # 使用动态 K
        stu_input = torch.cat([stu_emb_exp, k_dynamic], dim=-1)
        exer_input = torch.cat([exer_emb_exp, k_dynamic], dim=-1)
        
        # 计算 Utility
        s_vec = torch.sigmoid(self.f_sk(stu_input))
        e_vec = torch.sigmoid(self.f_ek(exer_input))
        diff = s_vec - e_vec
        concept_utility = self.f_se(diff).squeeze(-1) 
        
        # 4. 加权求和 (保持分母为 Q)
        weighted_utility = concept_utility * q_vector
        denominator = q_vector.sum(dim=1, keepdim=True) + 1e-8
        
        final_logit = weighted_utility.sum(dim=1, keepdim=True) / denominator
        
        return torch.sigmoid(final_logit).squeeze()