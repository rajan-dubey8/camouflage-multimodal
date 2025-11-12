"""
Visualization Utilities for Multimodal Camouflage Detection
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
import os

def plot_training_history(history, output_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved training history plot: {output_path}")

def plot_attention_heatmap(attention_weights, categories, output_path):
    """Plot attention heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(attention_weights, cmap='hot', aspect='auto')
    
    ax.set_xlabel('KG Categories', fontsize=12)
    ax.set_ylabel('RG Nodes (Regions)', fontsize=12)
    ax.set_title('Cross-Attention: RG → KG', fontsize=14, fontweight='bold')
    
    if len(categories) <= 20:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved attention heatmap: {output_path}")

def plot_comparison(image, pred_mask, gt_mask, output_path):
    """Plot image, prediction, and ground truth comparison"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction', fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(image)
    axes[3].imshow(pred_mask, alpha=0.5, cmap='hot')
    axes[3].set_title('Prediction Overlay', fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved comparison plot: {output_path}")

def plot_metrics_summary(metrics_dict, output_path):
    """Plot metrics summary as bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, color='skyblue', edgecolor='black', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 0.8:
            bar.set_color('green')
        elif val > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved metrics summary: {output_path}")