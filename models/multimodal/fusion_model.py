"""
Multimodal Fusion Model for Camouflage Detection
Combines Region Graph (RG) and Knowledge Graph (KG) embeddings

Fusion Strategies:
1. Cross-Attention Fusion: RG and KG attend to each other
2. Late Fusion: Simple concatenation + MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# ==================== CROSS-ATTENTION FUSION ====================
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention between RG and KG embeddings
    RG nodes attend to KG categories and vice versa
    """
    def __init__(self, rg_dim=128, kg_dim=128, hidden_dim=256, num_heads=8, dropout=0.3):
        super(CrossAttentionFusion, self).__init__()
        
        self.rg_dim = rg_dim
        self.kg_dim = kg_dim
        self.hidden_dim = hidden_dim
        
        # Project to same dimension if different
        self.rg_proj = nn.Linear(rg_dim, hidden_dim) if rg_dim != hidden_dim else nn.Identity()
        self.kg_proj = nn.Linear(kg_dim, hidden_dim) if kg_dim != hidden_dim else nn.Identity()
        
        # Cross-attention: RG queries, KG keys/values
        self.cross_attn_rg2kg = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: KG queries, RG keys/values
        self.cross_attn_kg2rg = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.ln_rg = nn.LayerNorm(hidden_dim)
        self.ln_kg = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn_rg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.ffn_kg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, rg_embeddings, kg_embeddings):
        """
        Args:
            rg_embeddings: [batch, num_rg_nodes, rg_dim] or [batch, rg_dim]
            kg_embeddings: [batch, num_kg_nodes, kg_dim] or [batch, kg_dim]
        
        Returns:
            fused: [batch, hidden_dim]
            attention_weights: dict with 'rg2kg' and 'kg2rg' attention maps
        """
        # Handle 2D inputs (graph-level embeddings)
        if rg_embeddings.dim() == 2:
            rg_embeddings = rg_embeddings.unsqueeze(1)
        if kg_embeddings.dim() == 2:
            kg_embeddings = kg_embeddings.unsqueeze(1)

        # --- FIX: collapse accidental 4D tensors ---
        def collapse_to_3d(tensor, name):
            if tensor.dim() == 3:
                return tensor
            if tensor.dim() == 4:
                B, a, b, d = tensor.shape
                if a == 1:
                    return tensor.squeeze(1)
                if b == 1:
                    return tensor.squeeze(2)
                return tensor.view(B, a * b, d)
            raise ValueError(f"{name} must be 2D/3D/4D tensor, got shape {tensor.shape}")

        rg_embeddings = collapse_to_3d(rg_embeddings, "rg_embeddings")
        kg_embeddings = collapse_to_3d(kg_embeddings, "kg_embeddings")

        # Project to same dimension
        rg_proj = self.rg_proj(rg_embeddings)  # [batch, num_rg, hidden_dim]
        kg_proj = self.kg_proj(kg_embeddings)  # [batch, num_kg, hidden_dim]

        # Cross-attention: RG attends to KG
        rg_attended, attn_rg2kg = self.cross_attn_rg2kg(
            query=rg_proj,
            key=kg_proj,
            value=kg_proj,
            need_weights=True,
            average_attn_weights=True
        )
        rg_attended = self.ln_rg(rg_proj + rg_attended)
        rg_attended = rg_attended + self.ffn_rg(rg_attended)

        # Cross-attention: KG attends to RG
        kg_attended, attn_kg2rg = self.cross_attn_kg2rg(
            query=kg_proj,
            key=rg_proj,
            value=rg_proj,
            need_weights=True,
            average_attn_weights=True
        )
        kg_attended = self.ln_kg(kg_proj + kg_attended)
        kg_attended = kg_attended + self.ffn_kg(kg_attended)

        # Global pooling
        rg_pooled = rg_attended.mean(dim=1)
        kg_pooled = kg_attended.mean(dim=1)

        # Concatenate and fuse
        combined = torch.cat([rg_pooled, kg_pooled], dim=-1)
        fused = self.fusion_layer(combined)

        attention_weights = {
            "rg2kg": attn_rg2kg,
            "kg2rg": attn_kg2rg
        }

        return fused, attention_weights

# ==================== LATE FUSION ====================
class LateFusion(nn.Module):
    """Simple late fusion: concatenate RG and KG embeddings"""
    def __init__(self, rg_dim=128, kg_dim=128, hidden_dim=256, dropout=0.3):
        super(LateFusion, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(rg_dim + kg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
    
    def forward(self, rg_embeddings, kg_embeddings):
        if rg_embeddings.dim() == 3:
            rg_embeddings = rg_embeddings.mean(dim=1)
        if kg_embeddings.dim() == 3:
            kg_embeddings = kg_embeddings.mean(dim=1)
        combined = torch.cat([rg_embeddings, kg_embeddings], dim=-1)
        fused = self.fusion(combined)
        return fused, None

# ==================== MULTIMODAL CAMOUFLAGE DETECTOR ====================
class MultimodalCamouflageDetector(nn.Module):
    """Complete multimodal model for camouflage detection"""
    def __init__(self, 
                 rg_dim=128, 
                 kg_dim=128, 
                 hidden_dim=256, 
                 num_heads=8,
                 fusion_type='cross_attention',
                 num_classes=2,
                 dropout=0.3):
        super(MultimodalCamouflageDetector, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                rg_dim=rg_dim,
                kg_dim=kg_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            final_dim = hidden_dim
        elif fusion_type == 'late':
            self.fusion = LateFusion(
                rg_dim=rg_dim,
                kg_dim=kg_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            final_dim = hidden_dim // 2
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.mask_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        self.instance_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        self.edge_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1)
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rg_embeddings, kg_embeddings, return_attention=False):
        fused, attention_weights = self.fusion(rg_embeddings, kg_embeddings)
        mask_out = self.mask_head(fused)
        instance_out = self.instance_head(fused)
        edge_out = self.edge_head(fused)
        score_out = self.score_head(fused)
        if return_attention:
            return mask_out, instance_out, edge_out, score_out, attention_weights
        else:
            return mask_out, instance_out, edge_out, score_out

# ==================== MODEL BUILDER ====================
def build_multimodal_model(config):
    model = MultimodalCamouflageDetector(
        rg_dim=config.get("rg_dim", 128),
        kg_dim=config.get("kg_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 8),
        fusion_type=config.get("fusion_type", "cross_attention"),
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.3)
    )
    return model

# ==================== TESTING ====================
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING MULTIMODAL FUSION MODEL")
    print("=" * 70)
    
    batch_size = 4
    num_rg_nodes = 500
    num_kg_nodes = 10
    rg_dim = 128
    kg_dim = 128
    
    rg_embeddings = torch.randn(batch_size, num_rg_nodes, rg_dim)
    kg_embeddings = torch.randn(batch_size, num_kg_nodes, kg_dim)
    
    model_ca = MultimodalCamouflageDetector(
        rg_dim=rg_dim,
        kg_dim=kg_dim,
        hidden_dim=256,
        num_heads=8,
        fusion_type="cross_attention"
    )
    
    mask_out, inst_out, edge_out, score_out, attn = model_ca(
        rg_embeddings, kg_embeddings, return_attention=True
    )
    
    print("✅ Cross-Attention Fusion passed with outputs:")
    print("Mask:", mask_out.shape, "Instance:", inst_out.shape, "Edge:", edge_out.shape, "Score:", score_out.shape)
    print("=" * 70)
