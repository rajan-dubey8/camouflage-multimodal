# ==================== ANALYZE PREDICTIONS WITHOUT GROUND TRUTH ====================
# This analyzes model behavior on any image, even without GT mask
# Usage: python analyze_no_gt.py --image path/to/image.jpg --model best_model.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from skimage.segmentation import slic
from skimage import graph, feature
from scipy import ndimage
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# ==================== MODEL ====================
class RegionGraphGNN(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=128, num_classes=2):
        super(RegionGraphGNN, self).__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        
        self.fc_shared = nn.Linear(hidden_channels, hidden_channels)
        
        self.fc_mask_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_mask_2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.fc_instance_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_instance_2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.fc_edge_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_edge_2 = nn.Linear(hidden_channels // 2, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr.numel() > 0 else None
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        x_shared = F.relu(self.fc_shared(x))
        x_shared = F.dropout(x_shared, p=0.2, training=self.training)
        
        x_mask = F.relu(self.fc_mask_1(x_shared))
        x_mask = F.dropout(x_mask, p=0.2, training=self.training)
        mask_out = self.fc_mask_2(x_mask)
        
        x_inst = F.relu(self.fc_instance_1(x_shared))
        x_inst = F.dropout(x_inst, p=0.2, training=self.training)
        instance_out = self.fc_instance_2(x_inst)
        
        x_edge = F.relu(self.fc_edge_1(x_shared))
        x_edge = F.dropout(x_edge, p=0.2, training=self.training)
        edge_out = self.fc_edge_2(x_edge)
        
        return mask_out, instance_out, edge_out

# ==================== GRAPH CREATION ====================
def create_region_graph(image, n_segments=500):
    image_for_slic = (image * 255).astype(np.uint8)
    segments = slic(image_for_slic, n_segments=n_segments, compactness=10, sigma=1)
    
    n_regions = segments.max() + 1
    node_features = []
    region_id_map = {}
    valid_region_count = 0
    
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    edges_canny = feature.canny(gray_image, sigma=2)
    
    for region_id in range(n_regions):
        region_mask_bool = (segments == region_id)
        if not region_mask_bool.any():
            continue
        
        region_pixels = image[region_mask_bool]
        if len(region_pixels) == 0:
            continue
        
        mean_color = region_pixels.mean(axis=0)
        std_color = region_pixels.std(axis=0)
        gray_pixels = gray_image[region_mask_bool]
        texture_mean = gray_pixels.mean()
        texture_std = gray_pixels.std()
        
        coords = np.argwhere(region_mask_bool)
        center_y = coords[:, 0].mean() / 256.0
        center_x = coords[:, 1].mean() / 256.0
        region_size = len(region_pixels) / (256 * 256)
        
        perimeter = np.sum(ndimage.binary_dilation(region_mask_bool) ^ region_mask_bool)
        area = region_mask_bool.sum()
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-10)
        
        edge_density = edges_canny[region_mask_bool].mean()
        
        dilated = ndimage.binary_dilation(region_mask_bool, iterations=2)
        neighbor_mask = dilated & ~region_mask_bool
        contrast = 0.0
        if neighbor_mask.any():
            neighbor_colors = image[neighbor_mask]
            contrast = np.linalg.norm(mean_color - neighbor_colors.mean(axis=0))
        
        local_variance = np.var(gray_pixels)
        
        features = np.concatenate([
            mean_color, std_color,
            [texture_mean], [texture_std],
            [center_x, center_y],
            [region_size], [compactness],
            [contrast], [edge_density], [local_variance]
        ])
        features = np.nan_to_num(features, nan=0.0)
        node_features.append(features)
        
        region_id_map[region_id] = valid_region_count
        valid_region_count += 1
    
    node_features = torch.FloatTensor(np.array(node_features))
    
    rag = graph.rag_mean_color(image_for_slic, segments)
    edge_index = []
    edge_weight = []
    
    for edge in rag.edges():
        i, j = edge
        if i in region_id_map and j in region_id_map:
            new_i = region_id_map[i]
            new_j = region_id_map[j]
            edge_index.extend([[new_i, new_j], [new_j, new_i]])
            
            color_diff = np.linalg.norm(node_features[new_i][:3] - node_features[new_j][:3])
            texture_diff = abs(node_features[new_i][6] - node_features[new_j][6])
            edge_diff = abs(node_features[new_i][12] - node_features[new_j][12])
            
            weight = np.exp(-color_diff / 0.15) * \
                    np.exp(-texture_diff / 0.08) * \
                    np.exp(-edge_diff / 0.1)
            
            edge_weight.extend([weight, weight])
    
    edge_index = torch.LongTensor(edge_index).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    edge_weight = torch.FloatTensor(edge_weight) if edge_weight else torch.zeros(0)
    
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1) if len(edge_weight) > 0 else torch.zeros((0, 1))
    )
    return graph_data, segments

# ==================== ANALYSIS WITHOUT GT ====================
def analyze_without_gt(image_path, model_path, output_dir='analysis_results'):
    """Analyze model predictions without ground truth"""
    
    print("="*80)
    print("🔬 MODEL BEHAVIOR ANALYSIS (No Ground Truth)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"\n[1/4] Loading model: {os.path.basename(model_path)}")
    model = RegionGraphGNN(in_channels=15, hidden_channels=128, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  ✓ Device: {device}")
    
    # Load image
    print(f"\n[2/4] Loading image: {os.path.basename(image_path)}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"  Original size: {original_size}")
    image = image.resize((256, 256))
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    img_np_normalized = image_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np_normalized * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create graph
    print(f"\n[3/4] Creating region graph...")
    graph_data, segments = create_region_graph(img_np, n_segments=500)
    print(f"  ✓ Regions: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
    
    # Get predictions
    print(f"\n[4/4] Running predictions...")
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        mask_out, _, _ = model(graph_data)
        
        # Get RAW logits
        logits = mask_out.cpu().numpy()
        
        # Get probabilities
        probs = F.softmax(mask_out, dim=1).cpu().numpy()
        
        # Get predictions
        preds = mask_out.argmax(dim=1).cpu().numpy()
    
    # Analyze probability distribution
    print(f"\n  📊 Prediction Statistics:")
    print(f"    ─────────────────────────────────")
    print(f"    Class 1 Probabilities:")
    print(f"      Mean:     {probs[:, 1].mean():.4f}")
    print(f"      Median:   {np.median(probs[:, 1]):.4f}")
    print(f"      Std:      {probs[:, 1].std():.4f}")
    print(f"      Min:      {probs[:, 1].min():.4f}")
    print(f"      Max:      {probs[:, 1].max():.4f}")
    print(f"    ─────────────────────────────────")
    
    # Check if model is biased
    bias_detected = False
    if probs[:, 1].mean() > 0.7:
        print(f"\n  ⚠️  MODEL BIAS DETECTED: POSITIVE CLASS")
        print(f"      Average probability: {probs[:, 1].mean():.3f}")
        print(f"      Model predicts 'camouflaged' too often!")
        bias_detected = True
    elif probs[:, 1].mean() < 0.3:
        print(f"\n  ⚠️  MODEL BIAS DETECTED: NEGATIVE CLASS")
        print(f"      Average probability: {probs[:, 1].mean():.3f}")
        print(f"      Model predicts 'not camouflaged' too often!")
        bias_detected = True
    else:
        print(f"\n  ✓ Model seems reasonably calibrated")
    
    # Predictions at different thresholds
    print(f"\n  🎯 Predictions at Different Thresholds:")
    print(f"    ─────────────────────────────────")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds_thresh = (probs[:, 1] > thresh).sum()
        pct = preds_thresh / len(probs) * 100
        print(f"    Threshold {thresh:.1f}: {preds_thresh:4d}/{len(probs)} regions ({pct:5.1f}%)")
    print(f"    ─────────────────────────────────")
    
    # Reconstruct masks at different thresholds
    n_regions = segments.max() + 1
    
    masks_at_thresholds = {}
    for thresh in [0.3, 0.5, 0.7]:
        mask = np.zeros((256, 256))
        for region_id in range(n_regions):
            if region_id < len(probs):
                region_mask = (segments == region_id)
                mask[region_mask] = (probs[region_id, 1] > thresh).astype(float)
        masks_at_thresholds[thresh] = mask
    
    # Create comprehensive visualization
    print(f"\n  💾 Creating visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Original and heatmap
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_np)
    ax1.set_title(f'Original Image\n{os.path.basename(image_path)}', 
                 fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(segments, cmap='nipy_spectral')
    ax2.set_title(f'Superpixel Regions\n{n_regions} regions', 
                 fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # Probability heatmap
    pred_mask_full = np.zeros((256, 256))
    for region_id in range(n_regions):
        if region_id < len(probs):
            pred_mask_full[segments == region_id] = probs[region_id, 1]
    
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(pred_mask_full, cmap='hot', vmin=0, vmax=1)
    ax3.set_title(f'Probability Heatmap\nMean: {pred_mask_full.mean():.3f}', 
                 fontweight='bold', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(img_np)
    ax4.imshow(pred_mask_full, alpha=0.6, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('Heatmap Overlay', fontweight='bold', fontsize=12)
    ax4.axis('off')
    
    # Row 2: Different thresholds
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(masks_at_thresholds[0.3], cmap='gray')
    coverage = masks_at_thresholds[0.3].sum() / (256*256) * 100
    ax5.set_title(f'Threshold = 0.3\nCoverage: {coverage:.1f}%', 
                 fontweight='bold', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(masks_at_thresholds[0.5], cmap='gray')
    coverage = masks_at_thresholds[0.5].sum() / (256*256) * 100
    ax6.set_title(f'Threshold = 0.5\nCoverage: {coverage:.1f}%', 
                 fontweight='bold', fontsize=12)
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(masks_at_thresholds[0.7], cmap='gray')
    coverage = masks_at_thresholds[0.7].sum() / (256*256) * 100
    ax7.set_title(f'Threshold = 0.7\nCoverage: {coverage:.1f}%', 
                 fontweight='bold', fontsize=12)
    ax7.axis('off')
    
    # Overlay with contours
    ax8 = plt.subplot(3, 4, 8)
    ax8.imshow(img_np)
    if masks_at_thresholds[0.5].sum() > 0:
        from skimage import measure
        contours = measure.find_contours(masks_at_thresholds[0.5], 0.5)
        for contour in contours:
            ax8.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    ax8.set_title('Contours (Threshold=0.5)', fontweight='bold', fontsize=12)
    ax8.axis('off')
    
    # Row 3: Analysis
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(probs[:, 1], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax9.axvline(probs[:, 1].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean={probs[:, 1].mean():.3f}')
    ax9.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax9.set_xlabel('Probability (Class 1)', fontsize=10)
    ax9.set_ylabel('Frequency', fontsize=10)
    ax9.set_title('Probability Distribution', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    ax10 = plt.subplot(3, 4, 10)
    # Cumulative distribution
    sorted_probs = np.sort(probs[:, 1])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax10.plot(sorted_probs, cumulative, linewidth=2, color='purple')
    ax10.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 threshold')
    ax10.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax10.set_xlabel('Probability', fontsize=10)
    ax10.set_ylabel('Cumulative Proportion', fontsize=10)
    ax10.set_title('Cumulative Distribution', fontweight='bold', fontsize=12)
    ax10.grid(alpha=0.3)
    ax10.legend()
    
    # Statistics panel
    ax11 = plt.subplot(3, 4, 11)
    stats_text = f"STATISTICS:\n\n"
    stats_text += f"Training Accuracy: 90%\n\n"
    stats_text += f"Probability Stats:\n"
    stats_text += f"  Mean: {probs[:, 1].mean():.4f}\n"
    stats_text += f"  Std:  {probs[:, 1].std():.4f}\n"
    stats_text += f"  Median: {np.median(probs[:, 1]):.4f}\n\n"
    stats_text += f"Coverage:\n"
    for thresh in [0.3, 0.5, 0.7]:
        cov = masks_at_thresholds[thresh].sum() / (256*256) * 100
        stats_text += f"  {thresh:.1f}: {cov:5.1f}%\n"
    
    ax11.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=11,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax11.axis('off')
    
    # Diagnosis panel
    ax12 = plt.subplot(3, 4, 12)
    diagnosis_text = f"DIAGNOSIS:\n\n"
    
    if bias_detected:
        if probs[:, 1].mean() > 0.7:
            diagnosis_text += "⚠️ POSITIVE BIAS\n\n"
            diagnosis_text += "Model predicts\n"
            diagnosis_text += "'camouflaged' for\n"
            diagnosis_text += "most regions.\n\n"
            diagnosis_text += "SOLUTIONS:\n"
            diagnosis_text += "• Use threshold > 0.7\n"
            diagnosis_text += "• Retrain with lower\n"
            diagnosis_text += "  class weights\n"
        else:
            diagnosis_text += "⚠️ NEGATIVE BIAS\n\n"
            diagnosis_text += "Model predicts\n"
            diagnosis_text += "'not camouflaged'\n"
            diagnosis_text += "for most regions.\n\n"
            diagnosis_text += "SOLUTIONS:\n"
            diagnosis_text += "• Use threshold < 0.3\n"
            diagnosis_text += "• Retrain with higher\n"
            diagnosis_text += "  class weights\n"
    else:
        diagnosis_text += "✓ WELL CALIBRATED\n\n"
        diagnosis_text += "Model probabilities\n"
        diagnosis_text += "are reasonably\n"
        diagnosis_text += "distributed.\n\n"
        diagnosis_text += "Use threshold = 0.5\n"
        diagnosis_text += "for detection.\n"
    
    color = 'salmon' if bias_detected else 'lightgreen'
    ax12.text(0.1, 0.5, diagnosis_text, ha='left', va='center', fontsize=11,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    ax12.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'analysis_{os.path.basename(image_path)}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Average Probability: {probs[:, 1].mean():.4f}")
    print(f"Regions analyzed: {len(probs)}")
    
    if bias_detected:
        print(f"\n⚠️  WARNING: Model shows bias in predictions")
        if probs[:, 1].mean() > 0.7:
            recommended_thresh = 0.7
            print(f"   Recommended threshold: {recommended_thresh} (instead of 0.5)")
            print(f"   This would give {(probs[:, 1] > recommended_thresh).sum()/len(probs)*100:.1f}% positive predictions")
        else:
            recommended_thresh = 0.3
            print(f"   Recommended threshold: {recommended_thresh} (instead of 0.5)")
    else:
        print(f"\n✓ Model appears well-calibrated")
        print(f"  Use standard threshold: 0.5")
    
    print("="*80)
    
    return probs[:, 1].mean(), bias_detected

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze model predictions without ground truth'
    )
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', '-m', type=str, default='best_model.pth',
                       help='Model path (default: best_model.pth)')
    parser.add_argument('--output', '-o', type=str, default='analysis_results',
                       help='Output directory (default: analysis_results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        exit(1)
    
    analyze_without_gt(args.image, args.model, args.output)