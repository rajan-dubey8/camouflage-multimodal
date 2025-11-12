"""
Fixed Multimodal Training with Aggressive Class Balancing
Key fixes:
1. Aggressive Focal Loss (alpha=0.75, gamma=3.0)
2. Heavy oversampling of minority class
3. F1 score as main metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import os
import argparse
import yaml
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
from PIL import Image
import cv2

from fusion_model import MultimodalCamouflageDetector, build_multimodal_model
from embedding_matcher import EmbeddingMatcher


# ==================== AGGRESSIVE FOCAL LOSS ====================

class AggressiveFocalLoss(nn.Module):
    """
    Aggressive Focal Loss with higher alpha and gamma
    Forces model to learn minority class
    """
    def __init__(self, alpha=0.75, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch] class indices
        """
        ce_loss = self.ce_loss(inputs, targets)
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Alpha weighting (give more weight to minority class)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal term
        focal_weight = (1 - pt) ** self.gamma
        
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ==================== SMART LABEL EXTRACTION (KEEP EXISTING) ====================

def extract_label_from_mask(mask_path, threshold=0.1):
    """Extract label from ground truth mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0, 0.0
    
    mask_norm = mask.astype(float) / 255.0
    mean_intensity = mask_norm.mean()
    non_zero_ratio = (mask > 10).sum() / mask.size
    
    # Edge detection
    edges = cv2.Canny(mask, 50, 150)
    edge_ratio = (edges > 0).sum() / mask.size
    
    # Complexity
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    complexity = len(contours)
    
    if mean_intensity > threshold and non_zero_ratio > 0.05:
        if edge_ratio < 0.02 or complexity > 10:
            label = 1
            confidence = min(mean_intensity * 2, 1.0)
        else:
            label = 1
            confidence = mean_intensity
    else:
        label = 0
        confidence = 1.0 - mean_intensity
    
    return label, confidence


# ==================== IMPROVED DATASET ====================

class SmartMultimodalDataset(Dataset):
    """Dataset with smart label extraction"""
    def __init__(self, matched_data, mask_dir, instance_dir, edge_dir, augment=False):
        self.matched_data = matched_data
        self.mask_dir = mask_dir
        self.instance_dir = instance_dir
        self.edge_dir = edge_dir
        self.augment = augment
        
        self.valid_samples = []
        label_counts = Counter()
        
        print("\n🔍 Extracting labels from ground truth masks...")
        
        for sample in tqdm(matched_data, desc="Processing samples"):
            img_name = sample['image_name']
            base_name = os.path.splitext(img_name)[0]
            
            mask_path = os.path.join(mask_dir, base_name + '.png')
            instance_path = os.path.join(instance_dir, base_name + '.png')
            edge_path = os.path.join(edge_dir, base_name + '.png')
            
            if os.path.exists(mask_path) and os.path.exists(instance_path) and os.path.exists(edge_path):
                label, confidence = extract_label_from_mask(mask_path)
                
                sample['label'] = label
                sample['confidence'] = confidence
                sample['mask_path'] = mask_path
                sample['edge_path'] = edge_path
                
                self.valid_samples.append(sample)
                label_counts[label] += 1
        
        print(f"\n✅ Processed {len(self.valid_samples)} samples")
        print(f"   Label distribution:")
        print(f"      Class 0 (Not Camouflaged): {label_counts[0]}")
        print(f"      Class 1 (Camouflaged): {label_counts[1]}")
        print(f"      Ratio: {label_counts[1]/max(label_counts[0], 1):.2f}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def get_labels(self):
        return [s['label'] for s in self.valid_samples]
    
    def get_aggressive_sample_weights(self):
        """
        AGGRESSIVE sampling weights
        Give 5x more weight to minority class
        """
        labels = [s['label'] for s in self.valid_samples]
        confidences = [s['confidence'] for s in self.valid_samples]
        
        class_counts = Counter(labels)
        majority_count = max(class_counts.values())
        
        # AGGRESSIVE: 5x oversampling of minority class
        class_weights = {}
        for c, count in class_counts.items():
            if c == 1:  # Minority class (camouflaged)
                class_weights[c] = (majority_count / count) * 5.0  # 5x boost
            else:
                class_weights[c] = 1.0
        
        sample_weights = [class_weights[labels[i]] * confidences[i] 
                         for i in range(len(labels))]
        
        return sample_weights
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        rg_node_emb = sample['rg_node_embeddings']
        kg_emb = sample['kg_embeddings']
        
        # Augmentation
        if self.augment and torch.rand(1) > 0.5:
            rg_node_emb = rg_node_emb + torch.randn_like(rg_node_emb) * 0.01
            kg_emb = kg_emb + torch.randn_like(kg_emb) * 0.01
        
        mask = np.array(Image.open(sample['mask_path']).convert('L'))
        edge_mask = np.array(Image.open(sample['edge_path']).convert('L'))
        
        return {
            'rg_node_emb': rg_node_emb,
            'kg_emb': kg_emb,
            'mask_label': sample['label'],
            'confidence': sample['confidence'],
            'edge_label': float(edge_mask.mean() > 10),
            'score_label': mask.mean() / 255.0,
            'image_name': sample['image_name']
        }


def collate_fn(batch):
    return batch


# ==================== TRAINING WITH F1 METRIC ====================

def calculate_f1_score(predictions, labels):
    """Calculate F1 score for both classes"""
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    
    # Class 1 metrics (minority class)
    precision_1 = tp / (tp + fp + 1e-8)
    recall_1 = tp / (tp + fn + 1e-8)
    f1_class_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + 1e-8)
    
    # Class 0 metrics
    precision_0 = tn / (tn + fn + 1e-8)
    recall_0 = tn / (tn + fp + 1e-8)
    f1_class_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + 1e-8)
    
    return {
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1,
        'f1_avg': (f1_class_0 + f1_class_1) / 2,
        'precision_1': precision_1,
        'recall_1': recall_1
    }


def train_epoch_fixed(model, dataloader, optimizer, device, epoch):
    """Training with aggressive focal loss"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Losses
    focal_loss_fn = AggressiveFocalLoss(alpha=0.75, gamma=3.0)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc=f"Training E{epoch}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        batch_loss = 0
        batch_preds = []
        batch_labels = []
        
        for sample in batch:
            rg_node = sample['rg_node_emb'].unsqueeze(0).to(device)
            kg = sample['kg_emb'].unsqueeze(0).to(device)
            
            mask_label = torch.tensor([sample['mask_label']], dtype=torch.long).to(device)
            edge_label = torch.tensor([sample['edge_label']], dtype=torch.float).to(device)
            score_label = torch.tensor([sample['score_label']], dtype=torch.float).to(device)
            
            # Forward
            mask_out, inst_out, edge_out, score_out = model(rg_node, kg)
            
            # 1. Main loss: Aggressive Focal Loss
            loss_mask = focal_loss_fn(mask_out, mask_label) * 3.0
            
            # 2. Instance loss
            loss_inst = F.cross_entropy(inst_out, mask_label) * 1.0
            
            # 3. Edge loss
            loss_edge = bce_loss_fn(edge_out.squeeze(1), edge_label) * 0.5
            
            # 4. Score loss
            loss_score = mse_loss_fn(score_out.squeeze(1), score_label) * 0.3
            
            loss = loss_mask + loss_inst + loss_edge + loss_score
            batch_loss += loss.item()
            loss.backward()
            
            # Store predictions
            pred = mask_out.argmax(dim=1).item()
            batch_preds.append(pred)
            batch_labels.append(mask_label.item())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += batch_loss
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)
        
        # Calculate running F1
        if len(all_preds) > 0:
            running_f1 = calculate_f1_score(
                torch.tensor(batch_preds), 
                torch.tensor(batch_labels)
            )
            pbar.set_postfix({
                'loss': batch_loss / len(batch),
                'f1_c1': f"{running_f1['f1_class_1']:.3f}",
                'f1_c0': f"{running_f1['f1_class_0']:.3f}"
            })
    
    # Final metrics
    avg_loss = total_loss / len(all_preds)
    f1_metrics = calculate_f1_score(torch.tensor(all_preds), torch.tensor(all_labels))
    
    return avg_loss, f1_metrics


def validate_fixed(model, dataloader, device):
    """Validation with F1 metrics"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            for sample in batch:
                rg_node = sample['rg_node_emb'].unsqueeze(0).to(device)
                kg = sample['kg_emb'].unsqueeze(0).to(device)
                mask_label = torch.tensor([sample['mask_label']], dtype=torch.long).to(device)
                
                mask_out, _, _, _ = model(rg_node, kg)
                
                loss = criterion(mask_out, mask_label)
                total_loss += loss.item()
                
                pred = mask_out.argmax(dim=1).item()
                all_preds.append(pred)
                all_labels.append(mask_label.item())
    
    avg_loss = total_loss / len(all_preds)
    f1_metrics = calculate_f1_score(torch.tensor(all_preds), torch.tensor(all_labels))
    
    # Also calculate per-class accuracy
    correct_0 = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 0)
    correct_1 = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 1)
    total_0 = sum(1 for l in all_labels if l == 0)
    total_1 = sum(1 for l in all_labels if l == 1)
    
    acc_0 = 100 * correct_0 / max(total_0, 1)
    acc_1 = 100 * correct_1 / max(total_1, 1)
    
    return avg_loss, f1_metrics, acc_0, acc_1


# ==================== MAIN TRAINING ====================

def train_multimodal_fixed(config):
    """Fixed training with aggressive balancing"""
    print("\n" + "="*80)
    print(" "*25 + "FIXED MULTIMODAL TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load embeddings
    print(f"\n📊 Loading embeddings...")
    matcher = EmbeddingMatcher(
        config['rg_embeddings_path'],
        config['kg_embeddings_path']
    )
    
    matched_data = matcher.create_matched_dataset(
        use_all_kg_categories=config['use_all_kg_categories']
    )
    
    # Create dataset
    print(f"\n📦 Creating dataset...")
    dataset = SmartMultimodalDataset(
        matched_data,
        config['mask_dir'],
        config['instance_dir'],
        config['edge_dir'],
        augment=True
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n   Train: {train_size} | Val: {val_size}")
    
    # AGGRESSIVE weighted sampler
    sample_weights = dataset.get_aggressive_sample_weights()
    train_weights = [sample_weights[i] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    print(f"\n   ✅ Using AGGRESSIVE oversampling (5x minority class)")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    
    # Build model
    print(f"\n🏗️  Building model...")
    model = build_multimodal_model(config['model'])
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training
    print(f"\n🏋️  Training for {config['epochs']} epochs...")
    print(f"   Main metric: F1 Score for Class 1 (Camouflaged)")
    print("="*80)
    
    best_f1_class_1 = 0
    patience_counter = 0
    max_patience = 15
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1_class_0': [], 'train_f1_class_1': [], 'train_f1_avg': [],
        'val_f1_class_0': [], 'val_f1_class_1': [], 'val_f1_avg': [],
        'val_acc_0': [], 'val_acc_1': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-"*80)
        
        # Train
        train_loss, train_f1 = train_epoch_fixed(model, train_loader, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_f1, val_acc_0, val_acc_1 = validate_fixed(model, val_loader, device)
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1_class_0'].append(train_f1['f1_class_0'])
        history['train_f1_class_1'].append(train_f1['f1_class_1'])
        history['train_f1_avg'].append(train_f1['f1_avg'])
        history['val_f1_class_0'].append(val_f1['f1_class_0'])
        history['val_f1_class_1'].append(val_f1['f1_class_1'])
        history['val_f1_avg'].append(val_f1['f1_avg'])
        history['val_acc_0'].append(val_acc_0)
        history['val_acc_1'].append(val_acc_1)
        
        # Print
        print(f"Train: Loss={train_loss:.4f} | F1_C0={train_f1['f1_class_0']:.3f} F1_C1={train_f1['f1_class_1']:.3f} F1_Avg={train_f1['f1_avg']:.3f}")
        print(f"Val:   Loss={val_loss:.4f} | F1_C0={val_f1['f1_class_0']:.3f} F1_C1={val_f1['f1_class_1']:.3f} F1_Avg={val_f1['f1_avg']:.3f}")
        print(f"       Acc_C0={val_acc_0:.1f}% Acc_C1={val_acc_1:.1f}%")
        print(f"       Precision_C1={val_f1['precision_1']:.3f} Recall_C1={val_f1['recall_1']:.3f}")
        
        # Save best based on F1 for Class 1 (minority class)
        if val_f1['f1_class_1'] > best_f1_class_1:
            best_f1_class_1 = val_f1['f1_class_1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1_class_1': val_f1['f1_class_1'],
                'val_f1_avg': val_f1['f1_avg'],
                'val_acc_0': val_acc_0,
                'val_acc_1': val_acc_1,
                'config': config
            }, os.path.join(config['checkpoint_dir'], 'multimodal_best_fixed.pth'))
            
            print(f"💾 Saved best model! (F1 Class 1: {val_f1['f1_class_1']:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\n⚠️  Early stopping after {patience_counter} epochs")
                break
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"Best F1 Score (Class 1): {best_f1_class_1:.3f}")
    print("="*80)
    
    # Save history
    with open(os.path.join(config['checkpoint_dir'], 'training_history_fixed.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    train_multimodal_fixed(config)


if __name__ == "__main__":
    main()