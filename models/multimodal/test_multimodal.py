"""
Test Multimodal Camouflage Detection Model (Fixed Version)
Compatible with the fixed training script and checkpoint format

Usage:
    python -m models.multimodal.test_multimodal_fixed --checkpoint models/multimodal/checkpoints/multimodal_best_fixed.pth --image <img> --output results
"""

# ==================== IMPORTS ====================

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from tqdm import tqdm
from collections import OrderedDict

# Local imports (run from repo root)
from models.multimodal.fusion_model import MultimodalCamouflageDetector, build_multimodal_model
from models.multimodal.embedding_matcher import EmbeddingMatcher
from models.region_graph.extract_rg_embeddings import extract_embeddings_from_image, RegionGraphGNN


# ==================== INFERENCE ====================
def load_multimodal_model(checkpoint_path, device):
    """Load trained multimodal model (supports both old and new checkpoint formats)"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = build_multimodal_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   ✓ Epoch: {checkpoint['epoch']}")
    print(f"   ✓ Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Handle both old format (val_acc) and new format (F1 scores)
    if 'val_f1_class_1' in checkpoint:
        print(f"   ✓ Val F1 (Class 1): {checkpoint['val_f1_class_1']:.3f}")
        print(f"   ✓ Val F1 (Avg): {checkpoint['val_f1_avg']:.3f}")
        print(f"   ✓ Val Acc (Class 0): {checkpoint['val_acc_0']:.1f}%")
        print(f"   ✓ Val Acc (Class 1): {checkpoint['val_acc_1']:.1f}%")
    elif 'val_acc' in checkpoint:
        print(f"   ✓ Val Acc: {checkpoint['val_acc']:.2f}%")
        if 'balanced_acc' in checkpoint:
            print(f"   ✓ Balanced Acc: {checkpoint['balanced_acc']:.2f}%")

    return model, config


def build_ordered_kg_tensor(kg_embeddings):
    """
    Ensure KG embeddings have a stable ordering and return:
      - kg_tensor: torch.Tensor [num_kg, dim]
      - ordered_categories: OrderedDict mapping category->embedding (sorted by key)
    """
    if isinstance(kg_embeddings, dict) or hasattr(kg_embeddings, "items"):
        ordered_keys = sorted(list(kg_embeddings.keys()))
        ordered = OrderedDict()
        kg_list = []
        for k in ordered_keys:
            v = kg_embeddings[k]
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            ordered[k] = v
            kg_list.append(v)
        kg_tensor = torch.stack(kg_list)  # [num_kg, dim]
        return kg_tensor, ordered
    else:
        # Already a tensor/list-like
        kg_tensor = torch.as_tensor(kg_embeddings)
        ordered = OrderedDict((f"cat_{i}", kg_tensor[i]) for i in range(kg_tensor.shape[0]))
        return kg_tensor, ordered


def predict_single_image(multimodal_model, rg_model, image_path, kg_embeddings_dict, device):
    """
    Run inference on a single image

    Returns:
        predictions: dict with mask, instance, edge, score predictions
        attention_weights: attention maps (if using cross-attention)
        kg_categories_ordered: OrderedDict of KG categories -> embedding (keeps label order)
    """
    # Extract RG embedding
    rg_node_emb, rg_graph_emb, graph_data, segments = extract_embeddings_from_image(
        rg_model, image_path, device, n_segments=500
    )

    # Build ordered KG tensor and keep categories mapping
    kg_tensor, kg_ordered = build_ordered_kg_tensor(kg_embeddings_dict)
    kg_emb = kg_tensor.unsqueeze(0).to(device)  # [1, num_kg, dim]

    # Prepare RG embedding
    rg_node_emb = rg_node_emb.unsqueeze(0).to(device)  # [1, num_nodes, dim]

    # Forward pass
    with torch.no_grad():
        outputs = multimodal_model(rg_node_emb, kg_emb, return_attention=True)
        
        # Unpack outputs robustly
        if len(outputs) == 5:
            mask_out, inst_out, edge_out, score_out, attn_weights = outputs
        else:
            mask_out, inst_out, edge_out, score_out = outputs
            attn_weights = None

        # Probabilities (safe)
        mask_prob = None
        if mask_out is not None:
            try:
                mask_prob = F.softmax(mask_out, dim=1)
            except Exception:
                mask_prob = torch.sigmoid(mask_out).unsqueeze(0)

        inst_prob = None
        if inst_out is not None:
            try:
                inst_prob = F.softmax(inst_out, dim=1)
            except Exception:
                inst_prob = torch.sigmoid(inst_out).unsqueeze(0)

        edge_prob = torch.sigmoid(edge_out) if edge_out is not None else torch.tensor([0.0])

    # Build predictions (move to CPU)
    mask_logits_cpu = mask_out.cpu() if mask_out is not None else None
    pred_mask_val = None
    if mask_logits_cpu is not None:
        try:
            pred_mask_val = int(mask_logits_cpu.argmax(dim=1).cpu().item())
        except Exception:
            pred_mask_val = int(torch.argmax(mask_logits_cpu).item())

    predictions = {
        'mask_logits': mask_logits_cpu,
        'mask_prob': mask_prob.cpu() if mask_prob is not None else None,
        'mask_pred': pred_mask_val,
        'instance_prob': inst_prob.cpu() if inst_prob is not None else None,
        'instance_pred': int(inst_out.argmax(dim=1).cpu().item()) if inst_out is not None else 0,
        'edge_prob': float(edge_prob.cpu().item()) if edge_out is not None else 0.0,
        'score': float(score_out.cpu().item()) if score_out is not None else 0.0,
        'segments': segments
    }

    return predictions, attn_weights, kg_ordered


# ==================== VISUALIZATION ====================
def visualize_prediction(image_path, predictions, attention_weights, kg_categories_ordered, output_path):
    """
    Visualize multimodal prediction results.

    CRITICAL CLASS MAPPING (matches training script):
      - Class 0 = NOT CAMOUFLAGED
      - Class 1 = CAMOUFLAGED
    
    This is the CORRECT mapping for the fixed training script!
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image.resize((256, 256))) / 255.0

    fig = plt.figure(figsize=(20, 10))

    # 1. Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image_np)
    ax1.set_title(f'Original Image\n{os.path.basename(image_path)}', fontweight='bold')
    ax1.axis('off')

    # 2. Superpixels
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(predictions['segments'], cmap='nipy_spectral')
    try:
        regions_count = int(predictions["segments"].max() + 1)
    except Exception:
        regions_count = None
    ax2.set_title(f'Superpixels\n{regions_count} regions' if regions_count is not None else "Superpixels", fontweight='bold')
    ax2.axis('off')

    # 3. Prediction info and confidence
    # CRITICAL: Class 1 = CAMOUFLAGED, Class 0 = NOT CAMOUFLAGED
    camo_index = 1  # Changed from 0 to 1!
    not_camo_index = 0  # Changed from 1 to 0!

    # Safe extraction of probabilities
    probs_tensor = predictions['mask_prob']
    if probs_tensor is not None:
        probs = probs_tensor
        # probs expected shape [1, 2]
        if probs.dim() == 2 and probs.shape[1] >= 2:
            not_camo_prob = float(probs[0, not_camo_index].item())
            camo_prob = float(probs[0, camo_index].item())
        elif probs.dim() == 1 and probs.shape[0] >= 2:
            not_camo_prob = float(probs[not_camo_index].item())
            camo_prob = float(probs[camo_index].item())
        else:
            # fallback
            flat = probs.flatten().cpu().numpy()
            not_camo_prob = float(flat[0]) if flat.size > 0 else 1.0
            camo_prob = 1.0 - not_camo_prob
    else:
        not_camo_prob = 1.0
        camo_prob = 0.0

    score = predictions.get('score', 0.0)
    pred_label = predictions.get('mask_pred', 0)  # default to not camo

    # Show image and text
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(image_np)

    # Interpretation: if pred_label == 1 => camouflaged
    if pred_label == camo_index:
        result_text = f"🔴 CAMOUFLAGED\nConfidence: {camo_prob:.2%}\nScore: {score:.3f}"
        color = 'red'
    else:
        result_text = f"🟢 NOT CAMOUFLAGED\nConfidence: {not_camo_prob:.2%}\nScore: {score:.3f}"
        color = 'green'

    ax3.text(0.5, -0.1, result_text, transform=ax3.transAxes,
             ha='center', fontsize=12, fontweight='bold', color=color,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.set_title('Prediction', fontweight='bold')
    ax3.axis('off')

    # 4. Attention weights (RG -> KG)
    if attention_weights is not None and 'rg2kg' in attention_weights:
        ax4 = plt.subplot(2, 4, 4)
        attn_rg2kg = attention_weights.get('rg2kg', None)
        if attn_rg2kg is not None:
            attn_rg2kg = torch.as_tensor(attn_rg2kg)
            if attn_rg2kg.dim() == 4:
                attn_rg2kg = attn_rg2kg[0].mean(dim=0).mean(dim=0)
            elif attn_rg2kg.dim() == 3:
                attn_rg2kg = attn_rg2kg.mean(dim=0).mean(dim=0)
            elif attn_rg2kg.dim() == 2:
                attn_rg2kg = attn_rg2kg.mean(dim=0)
            else:
                attn_rg2kg = attn_rg2kg.flatten()

            attn_rg2kg = attn_rg2kg.cpu().numpy()
            top_k = min(10, attn_rg2kg.shape[0])
            top_indices = np.argsort(attn_rg2kg)[-top_k:][::-1]
            cat_keys = list(kg_categories_ordered.keys())
            top_categories = [cat_keys[i] for i in top_indices]
            top_weights = attn_rg2kg[top_indices]

            ax4.barh(range(top_k), top_weights, color='skyblue')
            ax4.set_yticks(range(top_k))
            ax4.set_yticklabels(top_categories, fontsize=8)
            ax4.set_xlabel('Attention Weight')
            ax4.set_title('Top Attended KG Categories', fontweight='bold')
            ax4.invert_yaxis()

    # 5. Class probabilities (correct order: [Not Camo, Camo])
    ax5 = plt.subplot(2, 4, 5)
    ax5_vals = [not_camo_prob, camo_prob]
    ax5.bar(['Not Camouflaged', 'Camouflaged'], ax5_vals, color=['green', 'red'], alpha=0.7)
    ax5.set_ylabel('Probability')
    ax5.set_ylim([0, 1])
    ax5.set_title('Class Probabilities', fontweight='bold')
    ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

    # 6. Confidence meter
    ax6 = plt.subplot(2, 4, 6)
    confidence = max(camo_prob, not_camo_prob)
    colors = ['red' if confidence < 0.6 else 'orange' if confidence < 0.8 else 'green']
    ax6.barh([0], [confidence], color=colors, height=0.5)
    ax6.set_xlim([0, 1])
    ax6.set_yticks([])
    ax6.set_xlabel('Confidence')
    ax6.set_title(f'Model Confidence: {confidence:.1%}', fontweight='bold')
    ax6.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    ax6.axvline(x=0.8, color='black', linestyle='--', alpha=0.3)

    # 7. Edge prediction
    ax7 = plt.subplot(2, 4, 7)
    edge_prob = predictions.get('edge_prob', 0.0)
    ax7.bar(['No Edge', 'Edge'], [1-edge_prob, edge_prob], color=['gray', 'blue'], alpha=0.7)
    ax7.set_ylabel('Probability')
    ax7.set_ylim([0, 1])
    ax7.set_title('Edge Detection', fontweight='bold')

    # 8. Statistics panel
    ax8 = plt.subplot(2, 4, 8)
    stats_text = "STATISTICS\n\n"
    stats_text += f"Prediction: {'Camouflaged' if pred_label==camo_index else 'Not Camouflaged'}\n"
    stats_text += f"Camo Prob: {camo_prob:.2%}\n"
    stats_text += f"Not Camo Prob: {not_camo_prob:.2%}\n\n"
    stats_text += f"Instance Pred: {predictions.get('instance_pred', 0)}\n"
    stats_text += f"Edge Prob: {edge_prob:.2%}\n\n"
    stats_text += f"Score: {score:.3f}\n\n"
    stats_text += f"Regions: {predictions['segments'].max()+1}\n"

    ax8.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax8.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


# ==================== BATCH TESTING ====================
def test_image_directory(multimodal_model, rg_model, image_dir, kg_embeddings_dict, 
                         output_dir, device, max_images=None):
    """Test on multiple images and generate statistics"""
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n📊 Testing on {len(image_files)} images...")
    
    # Statistics
    results = []
    camo_count = 0
    not_camo_count = 0
    
    for img_file in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(image_dir, img_file)
        
        try:
            predictions, attention, kg_ordered = predict_single_image(
                multimodal_model, rg_model, img_path, kg_embeddings_dict, device
            )
            
            pred_label = predictions['mask_pred']
            probs = predictions['mask_prob']
            
            # Extract probabilities
            if probs is not None and probs.dim() == 2 and probs.shape[1] >= 2:
                not_camo_prob = float(probs[0, 0].item())
                camo_prob = float(probs[0, 1].item())
            else:
                not_camo_prob = 0.5
                camo_prob = 0.5
            
            results.append({
                'image': img_file,
                'prediction': 'Camouflaged' if pred_label == 1 else 'Not Camouflaged',
                'pred_label': pred_label,
                'camo_prob': camo_prob,
                'not_camo_prob': not_camo_prob,
                'score': predictions['score']
            })
            
            if pred_label == 1:
                camo_count += 1
            else:
                not_camo_count += 1
            
            # Save visualization
            output_path = os.path.join(output_dir, f'pred_{img_file}')
            visualize_prediction(img_path, predictions, attention, kg_ordered, output_path)
            
        except Exception as e:
            print(f"   ⚠️  Error processing {img_file}: {e}")
            continue
    
    # Save results
    results_path = os.path.join(output_dir, 'batch_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📈 Batch Testing Results:")
    print(f"   Total Images: {len(results)}")
    print(f"   Camouflaged: {camo_count} ({100*camo_count/len(results):.1f}%)")
    print(f"   Not Camouflaged: {not_camo_count} ({100*not_camo_count/len(results):.1f}%)")
    print(f"   Results saved: {results_path}")
    
    return results


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Test Multimodal Camouflage Detector (Fixed Version)')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to multimodal checkpoint (use multimodal_best_fixed.pth)')
    parser.add_argument('--rg-model', type=str, 
                       default='models/region_graph/best_model.pth', 
                       help='Path to RG model')
    parser.add_argument('--kg-embeddings', type=str, 
                       default='models/knowledge_graph/kg_embeddings/all_embeddings.pt', 
                       help='Path to KG embeddings')
    parser.add_argument('--image', type=str, default=None, 
                       help='Single image to test')
    parser.add_argument('--image-dir', type=str, default=None, 
                       help='Directory of images to test')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=None, 
                       help='Max number of images to test (for batch mode)')

    args = parser.parse_args()

    print("=" * 80)
    print(" " * 20 + "MULTIMODAL TESTING (FIXED VERSION)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")

    # Load models
    print(f"\n📂 Loading models...")
    multimodal_model, config = load_multimodal_model(args.checkpoint, device)

    print(f"\n   Loading RG model: {args.rg_model}")
    rg_model = RegionGraphGNN(in_channels=15, hidden_channels=128, num_classes=2)
    rg_model.load_state_dict(torch.load(args.rg_model, map_location=device))
    rg_model = rg_model.to(device)
    rg_model.eval()
    print(f"   ✓ RG model loaded")

    print(f"\n   Loading KG embeddings: {args.kg_embeddings}")
    kg_embeddings_raw = torch.load(args.kg_embeddings, map_location=device)
    print(f"   ✓ Loaded {len(kg_embeddings_raw)} KG categories")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Single image mode
    if args.image:
        print(f"\n🖼️  Testing single image: {args.image}")
        predictions, attention, kg_ordered = predict_single_image(
            multimodal_model, rg_model, args.image, kg_embeddings_raw, device
        )

        # Print results (correct mapping: Class 1 = Camouflaged)
        pred_label = predictions.get('mask_pred', 0)
        probs = predictions['mask_prob']
        
        if probs is not None and probs.dim() == 2 and probs.shape[1] >= 2:
            not_camo_prob = float(probs[0, 0].item())
            camo_prob = float(probs[0, 1].item())
        else:
            not_camo_prob = 0.5
            camo_prob = 0.5

        print(f"\n📊 Results:")
        print(f"   Prediction: {'🔴 CAMOUFLAGED' if pred_label==1 else '🟢 NOT CAMOUFLAGED'}")
        print(f"   Camouflaged Prob: {camo_prob:.2%}")
        print(f"   Not Camouflaged Prob: {not_camo_prob:.2%}")
        print(f"   Score: {predictions['score']:.3f}")

        output_path = os.path.join(args.output, f'prediction_{os.path.basename(args.image)}')
        visualize_prediction(args.image, predictions, attention, kg_ordered, output_path)

    # Batch mode
    elif args.image_dir:
        print(f"\n📁 Testing directory: {args.image_dir}")
        results = test_image_directory(
            multimodal_model, rg_model, args.image_dir, kg_embeddings_raw,
            args.output, device, args.max_images
        )

    else:
        print("\n❌ Error: Please provide either --image or --image-dir")
        return

    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()