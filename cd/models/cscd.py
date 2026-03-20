import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCDModel

class KANProjector(nn.Module):
    """
    KAN projection head: maps input features into a high-order interaction space.

    Idea:
    Uses sinusoidal and cosine basis functions to expand the feature manifold,
    enhancing nonlinear representation capability.
    """
    def __init__(self, in_dim, out_dim, num_basis=3):
        super().__init__()
        # Base linear path
        self.base_linear = nn.Linear(in_dim, out_dim)
        # Basis expansion weights
        self.basis_weights = nn.Parameter(torch.Tensor(out_dim, num_basis))
        # Frequency control parameters
        self.frequency1 = nn.Linear(in_dim, 1)
        self.frequency2 = nn.Linear(in_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)
        nn.init.normal_(self.basis_weights, std=0.02)  # small variance init

    def forward(self, x):
        base = self.base_linear(x)
        x_norm = torch.tanh(x)  # normalization to stabilize training

        # KAN Basis Expansion (SiLU, Sin, Cos)
        basis_out = (
            self.basis_weights[:, 0] * F.silu(base) +
            self.basis_weights[:, 1] * torch.sin(base * torch.pi * self.frequency1(x)) +
            self.basis_weights[:, 2] * torch.cos(base * torch.pi * self.frequency2(x))
        )

        # Residual connection
        return base + basis_out


# ==========================================
# 2. Core Module: MHB-KAN
# ==========================================
class GradientDetachedContext(nn.Module):
    """
    [Top-Tier Complex Version]

    Name: Multi-Head Bilinear KAN Interaction Network (MHB-KAN)

    Key Design Ideas:

    1. Tensor Fusion:
       Uses bilinear pooling to capture second-order interactions between
       semantic features (A) and knowledge features (K).

    2. Structural Constraint:
       The Q-matrix acts as an interaction mask to suppress semantic noise.

    3. Multi-Head Mechanism:
       Parallel modeling in multiple subspaces to improve robustness.

    4. Near-Zero Initialization:
       Initializes output layer with very small noise to enable smooth warm-up.
    """
    def __init__(self, emb_dim, n_concepts, a_dim=11, hidden_dim=64, num_heads=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # === A. Dual-stream KAN Projectors ===
        # Stream 1: Semantic stream (E + A)
        self.semantic_proj = KANProjector(emb_dim + a_dim, hidden_dim)

        # Stream 2: Knowledge stream (K)
        self.knowledge_proj = KANProjector(emb_dim, hidden_dim)

        # === B. Bilinear Interaction Core ===
        self.bilinear_dropout = nn.Dropout(0.01)

        # === C. Structure-aware Fusion ===
        self.q_proj = nn.Linear(1, hidden_dim)

        # === D. Output Generator ===
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        self._init_params()

    def _init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Near-zero initialization for stable warm start
        nn.init.normal_(self.out_proj[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, k_static, exer_emb, a_vector, q_vector):
        batch_size = k_static.size(0)
        n_concepts = k_static.size(1)

        # 1. Gradient Detach (protect main network)
        exer_context = exer_emb.detach()
        raw_semantic = torch.cat([exer_context, a_vector], dim=-1)

        # ==========================================
        # Step 1: KAN Projection
        # ==========================================
        s_vec = self.semantic_proj(raw_semantic).unsqueeze(1)
        k_vec = self.knowledge_proj(k_static)

        # ==========================================
        # Step 2: Multi-head split
        # ==========================================
        s_heads = s_vec.view(batch_size, 1, self.num_heads, self.head_dim)
        k_heads = k_vec.view(batch_size, n_concepts, self.num_heads, self.head_dim)

        # ==========================================
        # Step 3: Bilinear Interaction
        # ==========================================
        interaction = s_heads * k_heads

        # ==========================================
        # Step 4: Structural Injection
        # ==========================================
        q_mask = q_vector.view(batch_size, n_concepts, 1, 1)

        q_feat = self.q_proj(q_vector.unsqueeze(-1))
        q_heads = q_feat.view(batch_size, n_concepts, self.num_heads, self.head_dim)

        # Fusion: (interaction + structure) * mask
        fused_heads = (interaction + q_heads) * q_mask

        # ==========================================
        # Step 5: Aggregation & Output
        # ==========================================
        fused_feat = fused_heads.view(batch_size, n_concepts, self.hidden_dim)
        fused_feat = self.bilinear_dropout(fused_feat)

        delta_k = self.out_proj(fused_feat)

        return delta_k


# ==========================================
# 3. Main Model: KSCD
# ==========================================
class KSCD(BaseCDModel):
    def __init__(self, n_students, n_exercises, n_concepts, emb_dim=20, a_dim=11):
        super(KSCD, self).__init__()

        self.n_students = n_students
        self.n_exercises = n_exercises
        self.n_concepts = n_concepts
        self.emb_dim = emb_dim

        # === Embeddings ===
        self.student_emb = nn.Embedding(self.n_students, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.n_exercises, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.n_concepts, self.emb_dim))

        self.emb_dropout = nn.Dropout(0.2)
        nn.init.kaiming_normal_(self.knowledge_emb)

        # === Prediction layers ===
        self.f_sk = nn.Linear(self.emb_dim * 2, self.n_concepts)
        self.f_ek = nn.Linear(self.emb_dim * 2, self.n_concepts)
        self.f_se = nn.Linear(self.n_concepts, 1)

        # === Dynamic Module (MHB-KAN) ===
        self.context_module = GradientDetachedContext(
            emb_dim=emb_dim,
            n_concepts=n_concepts,
            a_dim=a_dim,
            hidden_dim=64,
            num_heads=4
        )

        # Fusion coefficient
        self.fusion_alpha = nn.Parameter(torch.tensor(0.1))

        self._init_params()

    def _init_params(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, exer_id, q_vector, a_vector=None):
        # Compatibility check
        if a_vector is None:
            raise ValueError("a_vector is missing! Please check Trainer inputs.")

        # 1. Embeddings
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(exer_id)

        stu_emb_main = self.emb_dropout(stu_emb)
        exer_emb_main = self.emb_dropout(exer_emb)

        batch_size = stu_id.size(0)

        # Static K
        k_static = self.knowledge_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # ==========================================
        # 2. Dynamic K (MHB-KAN)
        # ==========================================
        delta_k = self.context_module(
            k_static,
            exer_emb_main,
            a_vector,
            q_vector
        )

        alpha = F.softplus(self.fusion_alpha)
        k_dynamic = k_static + alpha * delta_k

        # ==========================================
        # 3. KSCD Core Logic
        # ==========================================
        stu_emb_exp = stu_emb_main.unsqueeze(1).repeat(1, self.n_concepts, 1)
        exer_emb_exp = exer_emb_main.unsqueeze(1).repeat(1, self.n_concepts, 1)

        stu_input = torch.cat([stu_emb_exp, k_dynamic], dim=-1)
        exer_input = torch.cat([exer_emb_exp, k_dynamic], dim=-1)

        s_vec = torch.sigmoid(self.f_sk(stu_input))
        e_vec = torch.sigmoid(self.f_ek(exer_input))

        diff = s_vec - e_vec
        concept_utility = self.f_se(diff).squeeze(-1)

        # Weighted aggregation (denominator = Q)
        weighted_utility = concept_utility * q_vector
        denominator = q_vector.sum(dim=1, keepdim=True) + 1e-8

        final_logit = weighted_utility.sum(dim=1, keepdim=True) / denominator

        return torch.sigmoid(final_logit).squeeze()
