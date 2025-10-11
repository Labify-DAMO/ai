# === Paste your actual MultiHead model definition here ===
# The API expects the model to return a dict:
# {
#   "bin":    Tensor[B, 2],            # optional if USE_BINARY_HEAD=True
#   "coarse": Tensor[B, n_coarse],
#   "fine":   Tensor[B, n_fine]
# }
#
# For convenience, below is a minimal stub to help you wire things up.
# Replace this with your real implementation from training.

# ============================================
# 4) 모델 정의 (ConvNeXt + Multi-Head)
# ============================================
import timm, torch, torch.nn as nn

class MultiHead(nn.Module):
    def __init__(self, backbone,
                 use_binary=True, n_coarse=0, n_fine=0,
                 use_supervised_contam=False, n_contam=0):
        super().__init__()

        # --- 백본 생성 ---
        self.backbone = backbone
        feat_dim = self.backbone.num_features

        # --- 헤드 정의 ---
        self.use_binary = use_binary
        self.use_supervised_contam = use_supervised_contam

        if use_binary:
            self.head_bin = nn.Linear(feat_dim, 2)
        if n_coarse > 0:
            self.head_coarse = nn.Linear(feat_dim, n_coarse)
        if n_fine > 0:
            self.head_fine = nn.Linear(feat_dim, n_fine)
        if use_supervised_contam and n_contam > 0:
            self.head_contam = nn.Linear(feat_dim, n_contam)

    def forward(self, x):
        feat = self.backbone(x)
        out = {}
        if hasattr(self, 'head_bin'):
            out['bin'] = self.head_bin(feat)
        if hasattr(self, 'head_coarse'):
            out['coarse'] = self.head_coarse(feat)
        if hasattr(self, 'head_fine'):
            out['fine'] = self.head_fine(feat)
        if hasattr(self, 'head_contam'):
            out['contam'] = self.head_contam(feat)
        return out
