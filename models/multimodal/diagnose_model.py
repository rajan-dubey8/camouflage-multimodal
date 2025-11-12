"""
Diagnose Model Bias
Quick script to check if your model has class imbalance issues
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
import sys

# ✅ Fix import path so it works when running directly from terminal
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


def diagnose_checkpoint(checkpoint_path):
    """Analyze checkpoint for bias issues"""
    print("="*80)
    print("MODEL DIAGNOSIS")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📋 Checkpoint Info:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 0):.4f}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
    
    if 'balanced_acc' in checkpoint:
        print(f"   Balanced Acc: {checkpoint['balanced_acc']:.2f}%")
        print(f"   Class 0 Acc: {checkpoint['class_acc'][0]:.2f}%")
        print(f"   Class 1 Acc: {checkpoint['class_acc'][1]:.2f}%")
    
    # Check model weights for bias
    state_dict = checkpoint['model_state_dict']
    
    # Check mask head final layer
    if 'mask_head.4.weight' in state_dict or 'mask_head.2.weight' in state_dict:
        try:
            mask_weight_key = 'mask_head.4.weight' if 'mask_head.4.weight' in state_dict else 'mask_head.2.weight'
            mask_weights = state_dict[mask_weight_key]
            
            print(f"\n🔍 Mask Head Weights Analysis:")
            print(f"   Shape: {mask_weights.shape}")
            print(f"   Class 0 weights norm: {mask_weights[0].norm().item():.4f}")
            print(f"   Class 1 weights norm: {mask_weights[1].norm().item():.4f}")
            ratio = (mask_weights[1].norm() / mask_weights[0].norm()).item()
            print(f"   Ratio (C1/C0): {ratio:.4f}")
            
            if abs(ratio - 1.0) > 0.5:
                print(f"   ⚠️  WARNING: Large weight imbalance detected!")
        except Exception as e:
            print(f"   ⚠️  Could not analyze weights: {e}")
    
    # Check if bias exists
    if 'mask_head.4.bias' in state_dict or 'mask_head.2.bias' in state_dict:
        try:
            mask_bias_key = 'mask_head.4.bias' if 'mask_head.4.bias' in state_dict else 'mask_head.2.bias'
            mask_bias = state_dict[mask_bias_key]
            
            print(f"\n   Mask Head Bias:")
            print(f"   Class 0 bias: {mask_bias[0].item():.4f}")
            print(f"   Class 1 bias: {mask_bias[1].item():.4f}")
            diff = (mask_bias[1] - mask_bias[0]).item()
            print(f"   Difference: {diff:.4f}")
            
            if abs(diff) > 2.0:
                print(f"   ⚠️  WARNING: Large bias difference! Model is biased toward one class.")
        except Exception as e:
            print(f"   ⚠️  Could not analyze bias: {e}")
    
    print("\n" + "="*80)


def test_model_on_dummy_data(checkpoint_path, num_samples=10):
    """Test model predictions on dummy data"""
    print("\n📊 Testing on Dummy Data:")
    
    # ✅ Fixed import to work when running directly
    from fusion_model import build_multimodal_model
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = build_multimodal_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Random embeddings
            rg_emb = torch.randn(1, 500, 128)  # Random RG
            kg_emb = torch.randn(1, 13, 128)   # Random KG
            
            mask_out, _, _, _ = model(rg_emb, kg_emb)
            
            probs = F.softmax(mask_out, dim=1)
            pred = mask_out.argmax(dim=1).item()
            
            predictions.append({
                'pred': pred,
                'prob_0': probs[0, 0].item(),
                'prob_1': probs[0, 1].item()
            })
    
    # Analyze predictions
    pred_counts = Counter([p['pred'] for p in predictions])
    avg_prob_0 = np.mean([p['prob_0'] for p in predictions])
    avg_prob_1 = np.mean([p['prob_1'] for p in predictions])
    
    print(f"   Predictions on {num_samples} random samples:")
    print(f"      Class 0 (Not Cam): {pred_counts[0]} times")
    print(f"      Class 1 (Camouflaged): {pred_counts[1]} times")
    print(f"   Average probabilities:")
    print(f"      P(Class 0): {avg_prob_0:.4f}")
    print(f"      P(Class 1): {avg_prob_1:.4f}")
    
    if pred_counts[0] == num_samples or pred_counts[1] == num_samples:
        print(f"\n   🚨 SEVERE BIAS DETECTED!")
        print(f"      Model always predicts class {0 if pred_counts[0] == num_samples else 1}")
        print(f"      Recommendation: RETRAIN with fixed labels and class balancing")
    elif abs(pred_counts[0] - pred_counts[1]) > num_samples * 0.7:
        print(f"\n   ⚠️  MODERATE BIAS DETECTED")
        print(f"      Recommendation: Retrain with class balancing")
    else:
        print(f"\n   ✓ Model appears balanced on random data")


def recommend_fixes():
    """Recommend fixes"""
    print("\n" + "="*80)
    print("RECOMMENDED FIXES:")
    print("="*80)
    print("""
1. ⚠️  CRITICAL: Fix Training Labels
   - Use filename-based labels (CAM vs NonCAM)
   - Not mask intensity-based labels
   
2. 🎯 Add Class Balancing
   - Use Focal Loss (alpha=0.25, gamma=2.0)
   - Use WeightedRandomSampler for training
   
3. 📊 Monitor Per-Class Accuracy
   - Track Class 0 and Class 1 accuracy separately
   - Use balanced accuracy as main metric
   
4. 🔄 Retrain from Scratch
   - Use the fixed train_multimodal.py script
   - Monitor class distribution during training
   
5. ✅ Validate on Both Classes
   - Test on both CAM and NonCAM images
   - Ensure balanced test set

Run: python train_multimodal.py --config configs/multimodal_config.yaml
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose model bias')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    diagnose_checkpoint(args.checkpoint)
    test_model_on_dummy_data(args.checkpoint, num_samples=20)
    recommend_fixes()
    
    print("\n" + "="*80)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*80)
