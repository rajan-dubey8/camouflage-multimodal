"""
Ablation Test for Multimodal Camouflage Detection
Tests the contribution of Knowledge Graph (KG) embeddings

Modes:
1. normal: Full model with real KG embeddings
2. zero_kg: Model with zero KG embeddings (tests if KG helps)
3. random_kg: Model with random KG embeddings (tests if KG structure matters)
"""

import torch
import torch.nn.functional as F
import os
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "models/multimodal/checkpoints/multimodal_best_fixed.pth"
KG_EMB_PATH = "models/knowledge_graph/kg_embeddings/all_embeddings.pt"
RG_EMB_PATH = "models/region_graph/rg_embeddings/all_rg_embeddings.pt"

# Folder containing test images
IMAGE_DIR = "test_images/"

# ==========================================================
# LOAD MULTIMODAL MODEL
# ==========================================================
print("\n" + "="*60)
print("🔍 MULTIMODAL CAMOUFLAGE DETECTION ABLATION TEST")
print("="*60)

print("\n[1/3] Loading multimodal checkpoint...")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"❌ Checkpoint not found: {CHECKPOINT}")

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

# Import model dynamically
import sys
sys.path.insert(0, os.path.abspath('.'))
from models.multimodal.fusion_model import build_multimodal_model

# Build model from config
if 'config' in checkpoint:
    config = checkpoint['config']
    model = build_multimodal_model(config['model'])
else:
    print("   ⚠️  No config in checkpoint, using default")
    from models.multimodal.fusion_model import MultimodalCamouflageDetector
    model = MultimodalCamouflageDetector(
        rg_dim=128, kg_dim=128, hidden_dim=256,
        num_heads=8, fusion_type='cross_attention'
    )

# Load weights
state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

print(f"   ✅ Model loaded successfully")
print(f"   📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"   📊 Val F1: {checkpoint.get('val_f1_class_1', checkpoint.get('val_acc', 'N/A'))}")

# ==========================================================
# LOAD REGION GRAPH (RG) EMBEDDINGS
# ==========================================================
print("\n[2/3] Loading precomputed RG embeddings...")
if not os.path.exists(RG_EMB_PATH):
    raise FileNotFoundError(f"❌ RG embeddings not found: {RG_EMB_PATH}")

rg_embeddings = torch.load(RG_EMB_PATH, map_location=DEVICE)

if isinstance(rg_embeddings, dict):
    print(f"   ✅ Loaded RG embeddings for {len(rg_embeddings)} images")
    sample_key = list(rg_embeddings.keys())[0]
    sample_data = rg_embeddings[sample_key]
    print(f"   📊 Sample structure:")
    print(f"      - node_embeddings: {sample_data['node_embeddings'].shape}")
    print(f"      - graph_embedding: {sample_data['graph_embedding'].shape}")
else:
    raise ValueError(f"❌ Unexpected RG embedding format: {type(rg_embeddings)}")

# ==========================================================
# LOAD KNOWLEDGE GRAPH (KG) EMBEDDINGS
# ==========================================================
print("\n[3/3] Loading Knowledge Graph (KG) embeddings...")
if not os.path.exists(KG_EMB_PATH):
    raise FileNotFoundError(f"❌ KG embedding file not found: {KG_EMB_PATH}")

kg_embeddings = torch.load(KG_EMB_PATH, map_location=DEVICE)

if isinstance(kg_embeddings, dict):
    kg_categories = sorted(kg_embeddings.keys())
    kg_list = [kg_embeddings[cat] for cat in kg_categories]
    kg_tensor = torch.stack(kg_list)
    while kg_tensor.dim() > 2:
        kg_tensor = kg_tensor.squeeze(1)
    kg_tensor = kg_tensor.to(DEVICE)
    print(f"   ✅ Loaded {len(kg_categories)} KG categories")
    print(f"   📊 KG tensor shape: {tuple(kg_tensor.shape)}")
    print(f"   📋 Categories: {kg_categories[:3]}...")
else:
    kg_tensor = torch.as_tensor(kg_embeddings).to(DEVICE)
    while kg_tensor.dim() > 2:
        kg_tensor = kg_tensor.squeeze()
    print(f"   ✅ KG tensor shape: {tuple(kg_tensor.shape)}")

kg_tensor = F.normalize(kg_tensor, dim=-1)

# ==========================================================
# DEFINE TEST MODES
# ==========================================================
print("\n" + "="*60)
print("📊 ABLATION MODES")
print("="*60)

MODES = {
    "normal": {
        "kg": kg_tensor,
        "desc": "Full model with real KG embeddings"
    },
    "zero_kg": {
        "kg": torch.zeros_like(kg_tensor),
        "desc": "Model with zero KG (tests if KG helps at all)"
    },
    "random_kg": {
        "kg": F.normalize(torch.randn_like(kg_tensor), dim=-1),
        "desc": "Model with random KG (tests if KG structure matters)"
    }
}

for mode_name, mode_data in MODES.items():
    print(f"   {mode_name:12s}: {mode_data['desc']}")

# ==========================================================
# FIXED: FIND MATCHING RG EMBEDDING
# ==========================================================
def find_rg_embedding(image_name, rg_embeddings):
    """
    Find matching RG embedding for an image.
    Falls back to a global average embedding if no match found.
    """
    base_name = os.path.splitext(image_name)[0]

    # Strategy 1: Exact match
    if image_name in rg_embeddings:
        return rg_embeddings[image_name]['node_embeddings']

    # Strategy 2: Base name match
    if base_name in rg_embeddings:
        return rg_embeddings[base_name]['node_embeddings']

    # Strategy 3: Try with common extensions
    for ext in ['.jpg', '.png', '.jpeg']:
        full_name = base_name + ext
        if full_name in rg_embeddings:
            return rg_embeddings[full_name]['node_embeddings']

    # Strategy 4: Partial match
    for key in rg_embeddings.keys():
        if base_name in key or key in base_name:
            print(f"   ⚠️  Using partial match: {key} for {image_name}")
            return rg_embeddings[key]['node_embeddings']

    # ✅ Fixed fallback (handles variable num_nodes safely)
    print(f"   ⚠️  No match found for {image_name}, using average RG embedding")

    per_image_means = []
    for data in rg_embeddings.values():
        node_emb = data['node_embeddings']
        per_image_means.append(node_emb.mean(dim=0))  # [128]

    avg_emb = torch.stack(per_image_means, dim=0).mean(dim=0, keepdim=True)  # [1, 128]
    avg_emb = avg_emb.unsqueeze(0)  # [1, 1, 128]
    return avg_emb.squeeze(0)

# ==========================================================
# TEST FUNCTION
# ==========================================================
@torch.no_grad()
def test_mode(mode_name, mode_data):
    kg_emb_tensor = mode_data['kg']
    preds, probs, confidences = [], [], []

    print(f"\n{'='*60}")
    print(f"🧪 Testing Mode: {mode_name.upper()}")
    print(f"   {mode_data['desc']}")
    print(f"{'='*60}")

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"❌ Test image directory not found: {IMAGE_DIR}")

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if len(image_files) == 0:
        raise FileNotFoundError(f"❌ No images found in: {IMAGE_DIR}")

    print(f"\n   Found {len(image_files)} test images\n")

    for i, img_name in enumerate(image_files, 1):
        rg_node_emb = find_rg_embedding(img_name, rg_embeddings)
        rg_node_emb = rg_node_emb.unsqueeze(0).to(DEVICE)  # [1, num_nodes, 128]
        kg_emb = kg_emb_tensor.unsqueeze(0).to(DEVICE)     # [1, num_kg, 128]

        try:
            mask_out, inst_out, edge_out, score_out = model(rg_node_emb, kg_emb)
        except Exception as e:
            print(f"   ❌ Error processing {img_name}: {e}")
            continue

        mask_probs = F.softmax(mask_out, dim=1)
        pred_label = mask_out.argmax(dim=1).item()
        camo_prob = mask_probs[0, 1].item()
        confidence = mask_probs.max().item()

        preds.append(pred_label)
        probs.append(camo_prob)
        confidences.append(confidence)

        status = "🔴 CAMO" if pred_label == 1 else "🟢 NOT"
        print(f"   [{i:2d}/{len(image_files)}] {img_name:30s} | {status} | "
              f"Prob={camo_prob:.3f} | Conf={confidence:.3f}")

    if len(probs) > 0:
        avg_prob = sum(probs) / len(probs)
        avg_conf = sum(confidences) / len(confidences)
        camo_count = sum(preds)

        print(f"\n   {'─'*56}")
        print(f"   📊 Summary:")
        print(f"      Total Images:       {len(probs)}")
        print(f"      Predicted Camo:     {camo_count} ({100*camo_count/len(probs):.1f}%)")
        print(f"      Predicted Not Camo: {len(probs)-camo_count} ({100*(len(probs)-camo_count)/len(probs):.1f}%)")
        print(f"      Avg Camo Prob:      {avg_prob:.3f}")
        print(f"      Avg Confidence:     {avg_conf:.3f}")
        print(f"   {'─'*56}")

    return {
        'predictions': preds,
        'probabilities': probs,
        'confidences': confidences,
        'avg_prob': avg_prob if len(probs) > 0 else 0,
        'avg_conf': avg_conf if len(probs) > 0 else 0,
        'camo_count': camo_count if len(probs) > 0 else 0
    }

# ==========================================================
# MAIN LOOP
# ==========================================================
if __name__ == "__main__":
    results = {}

    for mode_name, mode_data in MODES.items():
        results[mode_name] = test_mode(mode_name, mode_data)

    print("\n" + "="*60)
    print("📊 ABLATION RESULTS COMPARISON")
    print("="*60)

    print(f"\n{'Mode':<15s} | {'Camo%':<8s} | {'Avg Prob':<10s} | {'Avg Conf':<10s}")
    print("─" * 60)

    for mode_name, result in results.items():
        camo_pct = result['camo_count'] / len(result['predictions']) * 100 if result['predictions'] else 0
        print(f"{mode_name:<15s} | {camo_pct:>6.1f}% | "
              f"{result['avg_prob']:>8.3f} | {result['avg_conf']:>8.3f}")

    print("\n" + "="*60)
    print("💡 ANALYSIS")
    print("="*60)

    normal_prob = results['normal']['avg_prob']
    zero_prob = results['zero_kg']['avg_prob']
    random_prob = results['random_kg']['avg_prob']

    print(f"\n1. KG Contribution:")
    diff_zero = abs(normal_prob - zero_prob)
    if diff_zero > 0.1:
        print(f"   ✅ KG embeddings make a SIGNIFICANT difference!")
        print(f"      Normal vs Zero KG: {diff_zero:.3f} difference")
    elif diff_zero > 0.05:
        print(f"   ⚠️  KG embeddings make a MODERATE difference")
        print(f"      Normal vs Zero KG: {diff_zero:.3f} difference")
    else:
        print(f"   ❌ KG embeddings make MINIMAL difference")
        print(f"      Normal vs Zero KG: {diff_zero:.3f} difference")
        print(f"      → Model may not be using KG effectively!")

    print(f"\n2. KG Structure Importance:")
    diff_random = abs(normal_prob - random_prob)
    if diff_random > 0.1:
        print(f"   ✅ KG structure is IMPORTANT!")
        print(f"      Normal vs Random KG: {diff_random:.3f} difference")
    elif diff_random > 0.05:
        print(f"   ⚠️  KG structure has MODERATE importance")
        print(f"      Normal vs Random KG: {diff_random:.3f} difference")
    else:
        print(f"   ❌ KG structure has MINIMAL importance")
        print(f"      Normal vs Random KG: {diff_random:.3f} difference")
        print(f"      → Model may just be using KG as noise/regularization")

    print(f"\n3. Recommendations:")
    if diff_zero < 0.05:
        print(f"   • Consider increasing KG attention weight in fusion")
        print(f"   • Check if KG embeddings are meaningful")
        print(f"   • Try training with higher learning rate for KG pathway")
    elif diff_random < 0.05:
        print(f"   • Model uses KG but doesn't distinguish categories well")
        print(f"   • Try more diverse KG embeddings")
        print(f"   • Increase KG embedding dimension")
    else:
        print(f"   • Model is using KG effectively! ✅")
        print(f"   • Both presence and structure of KG matter")

    print("\n" + "="*60)
    print("✅ ABLATION TEST COMPLETE")
    print("="*60 + "\n")
