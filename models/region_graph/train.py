# ==================== KEY IMPROVEMENTS ====================
# 1. Better loss weighting for multi-task learning
# 2. Enhanced feature extraction with edge-aware features
# 3. Attention mechanism to focus on camouflaged regions
# 4.  ground truth labeling threshold
# 5. Better edge weight computation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage import graph, feature
from scipy import ndimage
import os
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# ====================  DATASET ====================
class CODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, instance_dir, edge_dir, transform=None, n_segments=500):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.instance_dir = instance_dir
        self.edge_dir = edge_dir
        self.transform = transform
        self.n_segments = n_segments
        
        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.images = []
        for img_name in all_images:
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, base_name + '.png')
            instance_path = os.path.join(instance_dir, base_name + '.png')
            edge_path = os.path.join(edge_dir, base_name + '.png')
            if os.path.exists(mask_path) and os.path.exists(instance_path) and os.path.exists(edge_path):
                self.images.append(img_name)
        
        print(f"Found {len(self.images)} valid image-mask-instance-edge quadruples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        instance_path = os.path.join(self.instance_dir, base_name + '.png')
        edge_path = os.path.join(self.edge_dir, base_name + '.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        instance_mask = Image.open(instance_path).convert('L')
        edge_mask = Image.open(edge_path).convert('L')
        
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))
        instance_mask = instance_mask.resize((256, 256))
        edge_mask = edge_mask.resize((256, 256))
        
        if self.transform:
            image_tensor = self.transform(image)
            mask = transforms.ToTensor()(mask)
            instance_mask = transforms.ToTensor()(instance_mask)
            edge_mask = transforms.ToTensor()(edge_mask)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Denormalize for feature extraction
        img_np_normalized = image_tensor.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np_normalized * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        mask_np = np.array(mask).squeeze()
        instance_np = np.array(instance_mask).squeeze()
        edge_np = np.array(edge_mask).squeeze()
        
        graph_data = self.create_region_graph(img_np, mask_np, instance_np, edge_np)
        
        return image_tensor, mask, instance_mask, edge_mask, graph_data, img_name
    
    def create_region_graph(self, image, mask, instance_mask, edge_mask):
        """
         Standard feature extraction with edge-aware features
        """
        mask = np.array(mask).squeeze()
        instance_mask = np.array(instance_mask).squeeze()
        edge_mask = np.array(edge_mask).squeeze()
        
        image_for_slic = (image * 255).astype(np.uint8)
        segments = slic(image_for_slic, n_segments=self.n_segments, compactness=10, sigma=1)
        
        n_regions = segments.max() + 1
        node_features = []
        node_labels = []
        instance_labels = []
        edge_labels = []
        region_id_map = {}
        valid_region_count = 0
        
        #  Compute edge-aware features
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
            
            #  Edge-aware features
            edge_density = edges_canny[region_mask_bool].mean()
            
            #  Boundary contrast with edge information
            dilated = ndimage.binary_dilation(region_mask_bool, iterations=2)
            neighbor_mask = dilated & ~region_mask_bool
            contrast = 0.0
            if neighbor_mask.any():
                neighbor_colors = image[neighbor_mask]
                contrast = np.linalg.norm(mean_color - neighbor_colors.mean(axis=0))
            
            #  Local texture variation
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
                [edge_density],          # 1: edge density (NEW)
                [local_variance]         # 1: local texture variance (NEW)
            ])
            features = np.nan_to_num(features, nan=0.0)
            node_features.append(features)
            
            #  Better ground truth labeling (threshold increased to 0.5)
            mask_pixels = mask[region_mask_bool]
            node_labels.append(1 if mask_pixels.mean() > 0.5 else 0)
            
            instance_pixels = instance_mask[region_mask_bool]
            instance_labels.append(1 if instance_pixels.mean() > 0.5 else 0)
            
            edge_pixels = edge_mask[region_mask_bool]
            edge_labels.append(1 if edge_pixels.mean() > 0.3 else 0)  # Keep lower for edges
            
            region_id_map[region_id] = valid_region_count
            valid_region_count += 1
        
        node_features = torch.FloatTensor(np.array(node_features))
        node_labels = torch.LongTensor(node_labels)
        instance_labels = torch.LongTensor(instance_labels)
        edge_labels = torch.FloatTensor(edge_labels)
        
        #  Adaptive edge weight computation
        rag = graph.rag_mean_color(image_for_slic, segments)
        edge_index = []
        edge_weight = []
        
        for edge in rag.edges():
            i, j = edge
            if i in region_id_map and j in region_id_map:
                new_i = region_id_map[i]
                new_j = region_id_map[j]
                edge_index.extend([[new_i, new_j], [new_j, new_i]])
                
                #  Multi-feature edge weight
                color_diff = np.linalg.norm(node_features[new_i][:3] - node_features[new_j][:3])
                texture_diff = abs(node_features[new_i][6] - node_features[new_j][6])
                edge_diff = abs(node_features[new_i][12] - node_features[new_j][12])  # NEW
                
                # Adaptive scaling based on feature statistics
                weight = np.exp(-color_diff / 0.15) * \
                        np.exp(-texture_diff / 0.08) * \
                        np.exp(-edge_diff / 0.1)
                
                edge_weight.extend([weight, weight])
        
        edge_index = torch.LongTensor(edge_index).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.FloatTensor(edge_weight) if edge_weight else torch.zeros(0)
        
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(1) if len(edge_weight) > 0 else torch.zeros((0, 1)),
            y=node_labels,
            instance_y=instance_labels,
            edge_y=edge_labels
        )
        return graph_data

# ==================== CUSTOM COLLATE ====================
def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    instance_masks = torch.stack([item[2] for item in batch])
    edge_masks = torch.stack([item[3] for item in batch])
    graph_data_list = [item[4] for item in batch]
    names = [item[5] for item in batch]
    graph_batch = Batch.from_data_list(graph_data_list)
    return images, masks, instance_masks, edge_masks, graph_batch, names

# ====================  GNN MODEL ====================
class RegionGraphGNN(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=128, num_classes=2):  # 15 features now
        super(RegionGraphGNN, self).__init__()
        
        #  Use GAT (Graph Attention) for first layer to learn important features
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        
        #  Separate feature extraction for each task
        self.fc_shared = nn.Linear(hidden_channels, hidden_channels)
        
        # Task-specific heads
        self.fc_mask_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_mask_2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.fc_instance_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_instance_2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.fc_edge_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_edge_2 = nn.Linear(hidden_channels // 2, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr.numel() > 0 else None
        
        # Layer 1 with attention
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 4
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Shared representation
        x_shared = F.relu(self.fc_shared(x))
        x_shared = F.dropout(x_shared, p=0.2, training=self.training)
        
        #  Task-specific branches
        # Mask prediction (main task)
        x_mask = F.relu(self.fc_mask_1(x_shared))
        x_mask = F.dropout(x_mask, p=0.2, training=self.training)
        mask_out = self.fc_mask_2(x_mask)
        
        # Instance prediction
        x_inst = F.relu(self.fc_instance_1(x_shared))
        x_inst = F.dropout(x_inst, p=0.2, training=self.training)
        instance_out = self.fc_instance_2(x_inst)
        
        # Edge prediction
        x_edge = F.relu(self.fc_edge_1(x_shared))
        x_edge = F.dropout(x_edge, p=0.2, training=self.training)
        edge_out = self.fc_edge_2(x_edge)
        
        return mask_out, instance_out, edge_out

# ====================  TRAINING ====================
def train_model(model, dataloader, val_loader=None, epochs=30, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    #  Better learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    #  Weighted multi-task loss
    # Weights: mask (most important) > instance > edge
    criterion_mask = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))
    criterion_instance = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(device))
    criterion_edge = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    
    #  Task weights for multi-task learning
    task_weights = {'mask': 2.0, 'instance': 1.0, 'edge': 0.5}
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_mask = 0
        total_mask = 0
        correct_instance = 0
        total_instance = 0
        
        for batch_idx, (images, masks, instance_masks, edge_masks, graph_data, names) in enumerate(dataloader):
            graph_data = graph_data.to(device)
            
            optimizer.zero_grad()
            mask_out, instance_out, edge_out = model(graph_data)
            
            #  Weighted multi-task loss
            loss_mask = criterion_mask(mask_out, graph_data.y) * task_weights['mask']
            loss_instance = criterion_instance(instance_out, graph_data.instance_y) * task_weights['instance']
            loss_edge = criterion_edge(edge_out.squeeze(), graph_data.edge_y.to(device)) * task_weights['edge']
            
            loss = loss_mask + loss_instance + loss_edge
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            pred_mask = mask_out.argmax(dim=1)
            correct_mask += (pred_mask == graph_data.y).sum().item()
            total_mask += graph_data.y.size(0)
            pred_instance = instance_out.argmax(dim=1)
            correct_instance += (pred_instance == graph_data.instance_y).sum().item()
            total_instance += graph_data.instance_y.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy_mask = 100 * correct_mask / total_mask
        accuracy_instance = 100 * correct_instance / total_instance
        
        if val_loader is not None:
            val_loss, val_acc_mask, val_acc_instance = validate_model(
                model, val_loader, criterion_mask, criterion_instance, criterion_edge, 
                device, task_weights
            )
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Mask: {accuracy_mask:.2f}% - Inst: {accuracy_instance:.2f}% | '
                  f'Val Loss: {val_loss:.4f} - Val Mask: {val_acc_mask:.2f}% - Val Inst: {val_acc_instance:.2f}%')
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'  → Best model saved! (Val Loss: {val_loss:.4f})')
        else:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Mask: {accuracy_mask:.2f}% - Inst: {accuracy_instance:.2f}%')
    
    return model

def validate_model(model, dataloader, criterion_mask, criterion_instance, criterion_edge, device, task_weights):
    model.eval()
    total_loss = 0
    correct_mask = 0
    total_mask = 0
    correct_instance = 0
    total_instance = 0
    
    with torch.no_grad():
        for images, masks, instance_masks, edge_masks, graph_data, names in dataloader:
            graph_data = graph_data.to(device)
            mask_out, instance_out, edge_out = model(graph_data)
            
            loss_mask = criterion_mask(mask_out, graph_data.y) * task_weights['mask']
            loss_instance = criterion_instance(instance_out, graph_data.instance_y) * task_weights['instance']
            loss_edge = criterion_edge(edge_out.squeeze(), graph_data.edge_y.to(device)) * task_weights['edge']
            loss = loss_mask + loss_instance + loss_edge
            
            total_loss += loss.item()
            pred_mask = mask_out.argmax(dim=1)
            correct_mask += (pred_mask == graph_data.y).sum().item()
            total_mask += graph_data.y.size(0)
            pred_instance = instance_out.argmax(dim=1)
            correct_instance += (pred_instance == graph_data.instance_y).sum().item()
            total_instance += graph_data.instance_y.size(0)
    
    return total_loss / len(dataloader), 100 * correct_mask / total_mask, 100 * correct_instance / total_instance

# ==================== MAIN ====================
if __name__ == "__main__":
    print("="*70)
    print(" CAMOUFLAGE OBJECT DETECTION - MULTI-TASK TRAINING")
    print("="*70)
    
    img_dir = "COD10K/images"
    mask_dir = "COD10K/gt_object"
    instance_dir = "COD10K/gt_instance"
    edge_dir = "COD10K/gt_edge"
    epochs = 30  # More epochs
    batch_size = 4
    learning_rate = 0.001  # Lower learning rate
    n_segments = 500
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n[1/5] Loading dataset...")
    dataset = CODDataset(img_dir, mask_dir, instance_dir, edge_dir, transform=transform, n_segments=n_segments)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    print(f"  ✓ Training: {train_size} | Validation: {val_size}")
    print(f"  ✓ Features: 15 (added edge density + local variance)")
    print(f"  ✓ Multi-task weights: Mask=2.0, Instance=1.0, Edge=0.5")
    
    print("\n[2/5] Initializing  model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegionGraphGNN(in_channels=15, hidden_channels=128, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Device: {device} | Parameters: {total_params:,}")
    
    print("\n[3/5] Training with multi-task learning...")
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)
    
    print("\n[4/5] Saving model...")
    torch.save(model.state_dict(), 'region_graph_model.pth')
    print("  ✓ Final model: 'region_graph_model.pth'")
    print("  ✓ Best model: 'best_model.pth'")
    
    print("\n[5/5] Training complete!")
    print("="*70)