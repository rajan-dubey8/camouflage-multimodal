"""
Extract Region Graph Embeddings for Multimodal Fusion
Extracts embeddings from trained RegionGraphGNN model with progress tracking

Usage:
    python extract_rg_embeddings.py --model best_model.pth --output rg_embeddings/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage import graph, feature
from scipy import ndimage
import os
import argparse
import json
from tqdm import tqdm
import time

# ==================== MODEL (Same as train.py) ====================
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
    
    def extract_node_embeddings(self, data):
        """
        Extract node-level embeddings (before classification heads)
        Returns: [num_nodes, 128] tensor
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr.numel() > 0 else None
        
        # Forward through GNN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Get shared embeddings (before task-specific heads)
        node_embeddings = F.relu(self.fc_shared(x))  # [num_nodes, 128]
        
        return node_embeddings
    
    def extract_graph_embedding(self, data):
        """
        Extract graph-level embedding (global pooling over nodes)
        Returns: [1, 128] tensor
        """
        node_embeddings = self.extract_node_embeddings(data)
        
        # Global mean pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
        graph_embedding = global_mean_pool(node_embeddings, batch)  # [1, 128]
        
        return graph_embedding

# ==================== REGION GRAPH CREATION ====================
def create_region_graph(image, n_segments=500):
    """
    Create region graph from image
    Returns: PyG Data object, segments
    """
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
        
        # Basic features
        mean_color = region_pixels.mean(axis=0)
        std_color = region_pixels.std(axis=0)
        gray_pixels = gray_image[region_mask_bool]
        texture_mean = gray_pixels.mean()
        texture_std = gray_pixels.std()
        
        # Position features
        coords = np.argwhere(region_mask_bool)
        center_y = coords[:, 0].mean() / 256.0
        center_x = coords[:, 1].mean() / 256.0
        region_size = len(region_pixels) / (256 * 256)
        
        # Shape features
        perimeter = np.sum(ndimage.binary_dilation(region_mask_bool) ^ region_mask_bool)
        area = region_mask_bool.sum()
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-10)
        
        # Edge-aware features
        edge_density = edges_canny[region_mask_bool].mean()
        
        # Boundary contrast
        dilated = ndimage.binary_dilation(region_mask_bool, iterations=2)
        neighbor_mask = dilated & ~region_mask_bool
        contrast = 0.0
        if neighbor_mask.any():
            neighbor_colors = image[neighbor_mask]
            contrast = np.linalg.norm(mean_color - neighbor_colors.mean(axis=0))
        
        # Local texture variation
        local_variance = np.var(gray_pixels)
        
        features = np.concatenate([
            mean_color,              # 3: RGB mean
            std_color,               # 3: RGB std
            [texture_mean],          # 1: texture mean
            [texture_std],           # 1: texture std
            [center_x, center_y],    # 2: position
            [region_size],           # 1: size
            [compactness],           # 1: shape
            [contrast],              # 1: boundary contrast
            [edge_density],          # 1: edge density
            [local_variance]         # 1: local texture variance
        ])
        features = np.nan_to_num(features, nan=0.0)
        node_features.append(features)
        
        region_id_map[region_id] = valid_region_count
        valid_region_count += 1
    
    node_features = torch.FloatTensor(np.array(node_features))
    
    # Create edges
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

# ==================== EMBEDDING EXTRACTION ====================
def extract_embeddings_from_image(model, image_path, device, n_segments=500):
    """
    Extract embeddings from a single image
    
    Returns:
        node_embeddings: [num_nodes, 128] - per-region embeddings
        graph_embedding: [1, 128] - global image embedding
        graph_data: PyG Data object
        segments: superpixel segmentation
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    # Denormalize for graph creation
    img_np_normalized = image_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np_normalized * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create region graph
    graph_data, segments = create_region_graph(img_np, n_segments=n_segments)
    graph_data = graph_data.to(device)
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings = model.extract_node_embeddings(graph_data)  # [num_nodes, 128]
        graph_embedding = model.extract_graph_embedding(graph_data)  # [1, 128]
    
    return node_embeddings.cpu(), graph_embedding.cpu(), graph_data.cpu(), segments

def format_time(seconds):
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def batch_extract_embeddings(model, image_dir, output_dir, device, n_segments=500, max_images=None):
    """
    Extract embeddings for all images in a directory with progress tracking
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("REGION GRAPH EMBEDDING EXTRACTION")
    print("="*80)
    
    # Get image files
    all_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(all_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    total_images = len(image_files)
    
    print(f"\n📊 Configuration:")
    print(f"   Input directory:  {image_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Total images:     {total_images}")
    print(f"   Segments/image:   {n_segments}")
    print(f"   Device:           {device}")
    
    # Estimate time per image (warmup)
    print(f"\n⏱️  Estimating processing time...")
    warmup_start = time.time()
    sample_img = os.path.join(image_dir, image_files[0])
    _, _, _, _ = extract_embeddings_from_image(model, sample_img, device, n_segments)
    time_per_image = time.time() - warmup_start
    
    estimated_total = time_per_image * total_images
    print(f"   Time per image:   ~{time_per_image:.2f}s")
    print(f"   Estimated total:  ~{format_time(estimated_total)}")
    
    all_embeddings = {}
    summary = {
        'total_images': total_images,
        'embedding_dim': 128,
        'n_segments': n_segments,
        'model_path': None,
        'processing_time': None,
        'images': {}
    }
    
    failed_images = []
    successful = 0
    
    # Main progress bar
    start_time = time.time()
    
    pbar = tqdm(
        image_files, 
        desc="🔄 Extracting embeddings",
        unit="img",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for idx, img_name in enumerate(pbar):
        try:
            img_path = os.path.join(image_dir, img_name)
            
            # Extract embeddings
            node_emb, graph_emb, graph_data, segments = extract_embeddings_from_image(
                model, img_path, device, n_segments
            )
            
            # Save individual embedding
            base_name = os.path.splitext(img_name)[0]
            save_path = os.path.join(output_dir, f"{base_name}_embedding.pt")
            
            torch.save({
                'image_name': img_name,
                'node_embeddings': node_emb,          # [num_nodes, 128]
                'graph_embedding': graph_emb,         # [1, 128]
                'num_nodes': node_emb.shape[0],
                'embedding_dim': node_emb.shape[1],
                'segments': segments,
                'graph_data': graph_data
            }, save_path)
            
            # Store in all_embeddings dict
            all_embeddings[img_name] = {
                'node_embeddings': node_emb,
                'graph_embedding': graph_emb,
                'num_nodes': node_emb.shape[0]
            }
            
            # Update summary
            summary['images'][img_name] = {
                'num_nodes': node_emb.shape[0],
                'embedding_path': save_path,
                'node_embedding_shape': list(node_emb.shape),
                'graph_embedding_shape': list(graph_emb.shape)
            }
            
            successful += 1
            
            # Update progress bar with current stats
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining_imgs = total_images - (idx + 1)
            eta = avg_time * remaining_imgs
            
            pbar.set_postfix({
                'success': successful,
                'failed': len(failed_images),
                'nodes': node_emb.shape[0],
                'ETA': format_time(eta)
            })
            
        except Exception as e:
            failed_images.append((img_name, str(e)))
            pbar.set_postfix({
                'success': successful,
                'failed': len(failed_images),
                'last_error': str(e)[:20]
            })
            continue
    
    pbar.close()
    
    # Calculate final stats
    total_time = time.time() - start_time
    avg_time_per_image = total_time / successful if successful > 0 else 0
    
    # Save all embeddings
    print(f"\n💾 Saving combined files...")
    all_embeddings_path = os.path.join(output_dir, 'all_rg_embeddings.pt')
    torch.save(all_embeddings, all_embeddings_path)
    print(f"   ✓ Saved: {all_embeddings_path}")
    
    # Update and save summary
    summary['processing_time'] = {
        'total_seconds': total_time,
        'total_formatted': format_time(total_time),
        'avg_per_image': avg_time_per_image,
        'successful_images': successful,
        'failed_images': len(failed_images)
    }
    
    summary_path = os.path.join(output_dir, 'embedding_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Saved: {summary_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 EXTRACTION STATISTICS")
    print("="*80)
    print(f"✅ Successful:        {successful}/{total_images} ({successful/total_images*100:.1f}%)")
    print(f"❌ Failed:            {len(failed_images)}")
    print(f"⏱️  Total time:        {format_time(total_time)}")
    print(f"⚡ Avg per image:     {avg_time_per_image:.2f}s")
    print(f"📦 Embedding dim:     128")
    
    if successful > 0:
        avg_nodes = np.mean([v['num_nodes'] for v in all_embeddings.values()])
        min_nodes = min([v['num_nodes'] for v in all_embeddings.values()])
        max_nodes = max([v['num_nodes'] for v in all_embeddings.values()])
        
        print(f"\n🔢 Region Statistics:")
        print(f"   Average nodes:     {avg_nodes:.1f}")
        print(f"   Min nodes:         {min_nodes}")
        print(f"   Max nodes:         {max_nodes}")
    
    if failed_images:
        print(f"\n⚠️  Failed Images:")
        for img_name, error in failed_images[:5]:  # Show first 5
            print(f"   - {img_name}: {error[:50]}")
        if len(failed_images) > 5:
            print(f"   ... and {len(failed_images) - 5} more")
    
    print(f"\n📁 Output Files:")
    print(f"   {output_dir}/")
    print(f"   ├── all_rg_embeddings.pt      ({os.path.getsize(all_embeddings_path) / (1024*1024):.1f} MB)")
    print(f"   ├── embedding_summary.json")
    print(f"   └── {successful} individual embedding files")
    
    return all_embeddings, summary

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description='Extract Region Graph embeddings with progress tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract all embeddings
  python extract_rg_embeddings.py --model best_model.pth --image-dir COD10K/images
  
  # Process first 100 images only
  python extract_rg_embeddings.py --model best_model.pth --image-dir COD10K/images --max-images 100
  
  # Single image
  python extract_rg_embeddings.py --model best_model.pth --single-image test.jpg
        '''
    )
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained RG model (default: best_model.pth)')
    parser.add_argument('--image-dir', type=str, default='COD10K/images',
                       help='Directory containing images (default: COD10K/images)')
    parser.add_argument('--output', type=str, default='rg_embeddings',
                       help='Output directory for embeddings (default: rg_embeddings)')
    parser.add_argument('--n-segments', type=int, default=500,
                       help='Number of superpixel segments (default: 500)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Max number of images to process (default: all)')
    parser.add_argument('--single-image', type=str, default=None,
                       help='Extract embedding for a single image')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" " * 20 + "REGION GRAPH EMBEDDING EXTRACTOR")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Load model
    print(f"\n📂 Loading model...")
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    model = RegionGraphGNN(in_channels=15, hidden_channels=128, num_classes=2)
    
    print(f"   Reading: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded successfully")
    print(f"   Parameters: {total_params:,}")
    
    # Single image mode
    if args.single_image:
        print(f"\n🖼️  Single Image Mode")
        print(f"   Image: {args.single_image}")
        
        if not os.path.exists(args.single_image):
            print(f"❌ Error: Image not found: {args.single_image}")
            return
        
        start_time = time.time()
        
        node_emb, graph_emb, graph_data, segments = extract_embeddings_from_image(
            model, args.single_image, device, args.n_segments
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Extraction complete! ({elapsed:.2f}s)")
        print(f"   Node embeddings:  {node_emb.shape}")
        print(f"   Graph embedding:  {graph_emb.shape}")
        print(f"   Number of regions: {node_emb.shape[0]}")
        
        # Save
        base_name = os.path.splitext(os.path.basename(args.single_image))[0]
        os.makedirs(args.output, exist_ok=True)
        save_path = os.path.join(args.output, f"{base_name}_embedding.pt")
        
        torch.save({
            'image_name': os.path.basename(args.single_image),
            'node_embeddings': node_emb,
            'graph_embedding': graph_emb,
            'num_nodes': node_emb.shape[0],
            'segments': segments,
            'graph_data': graph_data
        }, save_path)
        
        print(f"\n💾 Saved to: {save_path}")
        
    # Batch mode
    else:
        if not os.path.exists(args.image_dir):
            print(f"❌ Error: Image directory not found: {args.image_dir}")
            return
        
        all_embeddings, summary = batch_extract_embeddings(
            model, args.image_dir, args.output, device, 
            args.n_segments, args.max_images
        )
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()