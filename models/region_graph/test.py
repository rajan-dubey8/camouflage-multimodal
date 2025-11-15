# ==================== SIMPLE CAMOUFLAGE DETECTION TEST ====================
# This script tests ONLY the mask prediction (camouflage detection)
# Usage: python analyze_predictions.py --image ../../test_images/img15.jpg --model region_graph_model.pth
# Run Test : python test.py --image ../../test_images/img15.jpg --model region_graph_model.pth

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
    """Create region graph with 15 features"""
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

# ==================== PREDICTION ====================
def detect_camouflage(image_path, model_path, output_dir='results', mask_path=None):
    """Detect camouflaged objects in an image"""
    
    print("="*80)
    print("🎯 CAMOUFLAGED OBJECT DETECTION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"\n[1/4] Loading model...")
    model = RegionGraphGNN(in_channels=15, hidden_channels=128, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model: {os.path.basename(model_path)}")
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
    
    # Denormalize for visualization
    img_np_normalized = image_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np_normalized * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create graph
    print(f"\n[3/4] Creating region graph...")
    graph_data, segments = create_region_graph(img_np, n_segments=500)
    print(f"  ✓ Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
    
    # Predict
    print(f"\n[4/4] Detecting camouflage...")
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        mask_out, _, _ = model(graph_data)
        mask_probs = F.softmax(mask_out, dim=1)[:, 1].cpu().numpy()
    
    # Reconstruct full mask
    pred_mask = np.zeros((256, 256))
    for region_id in range(segments.max() + 1):
        if region_id < len(mask_probs):
            pred_mask[segments == region_id] = mask_probs[region_id]
    
    # Calculate statistics
    mean_score = pred_mask.mean()
    max_score = pred_mask.max()
    coverage = (pred_mask > 0.5).sum() / pred_mask.size * 100
    
    print(f"\n  📊 Detection Results:")
    print(f"    Mean score: {mean_score:.4f}")
    print(f"    Max score: {max_score:.4f}")
    print(f"    Coverage (>0.5): {coverage:.2f}%")
    
    # Classification
    if mean_score > 0.35:
        classification = "🔴 HIGHLY CAMOUFLAGED"
        color = 'red'
    elif mean_score > 0.20:
        classification = "🟡 MODERATELY CAMOUFLAGED"
        color = 'orange'
    elif mean_score > 0.10:
        classification = "🟠 SLIGHTLY CAMOUFLAGED"
        color = 'yellow'
    else:
        classification = "🟢 NOT CAMOUFLAGED"
        color = 'green'
    
    print(f"\n  {classification}")
    
    # Evaluate if GT available
    metrics = None
    if mask_path and os.path.exists(mask_path):
        gt_mask = np.array(Image.open(mask_path).convert('L').resize((256, 256))) / 255.0
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        
        iou = intersection / (union + 1e-8)
        dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
        
        tp = intersection
        fp = np.sum(pred_binary) - intersection
        fn = np.sum(gt_binary) - intersection
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        metrics = {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall, 'f1': f1}
        
        print(f"\n  📈 Performance Metrics:")
        print(f"    IoU: {iou:.4f}")
        print(f"    Dice: {dice:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
    
    # Visualize
    print(f"\n  💾 Saving results...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Original Image\n{os.path.basename(image_path)}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(segments, cmap='nipy_spectral')
    axes[0, 1].set_title(f'Superpixel Regions\n({segments.max()+1} regions)', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Camouflage Heatmap\nMean: {mean_score:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, label='Probability')
    
    # Row 2
    axes[1, 0].imshow(img_np)
    axes[1, 0].imshow(pred_mask, alpha=0.6, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Detection Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    pred_binary = (pred_mask > 0.5).astype(float)
    axes[1, 1].imshow(pred_binary, cmap='gray')
    axes[1, 1].set_title(f'Binary Mask (>0.5)\nCoverage: {coverage:.1f}%', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Final overlay with contours
    axes[1, 2].imshow(img_np)
    if pred_binary.sum() > 0:
        from skimage import measure
        contours = measure.find_contours(pred_binary, 0.5)
        for contour in contours:
            axes[1, 2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    axes[1, 2].set_title(f'{classification}\nScore: {mean_score:.3f}', 
                        fontsize=14, fontweight='bold', color=color)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'detection_{os.path.basename(image_path)}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Save mask
    mask_save_path = os.path.join(output_dir, f'mask_{os.path.basename(image_path)}')
    mask_img = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(mask_save_path)
    print(f"  ✓ Saved mask: {mask_save_path}")
    
    print("\n" + "="*80)
    print("✅ DETECTION COMPLETE")
    print("="*80)
    
    return pred_mask, mean_score, classification, metrics

# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect camouflaged objects in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python simple_test.py --image test.jpg
  
  # With ground truth for evaluation
  python simple_test.py --image img.jpg --mask gt_mask.png
  
  # Specify model and output directory
  python simple_test.py --image img.jpg --model best_model.pth --output results/
        '''
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', '-m', type=str, default='best_model.pth',
                       help='Model path (default: best_model.pth)')
    parser.add_argument('--mask', type=str, default=None,
                       help='Ground truth mask path (optional)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        exit(1)
    
    detect_camouflage(args.image, args.model, args.output, args.mask)