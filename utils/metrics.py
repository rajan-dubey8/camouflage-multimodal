"""
Evaluation Metrics for Camouflage Detection
IoU, Dice, Precision, Recall, F1, MAE
"""

import numpy as np
import torch

def calculate_iou(pred, gt, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    iou = intersection / (union + 1e-8)
    return iou

def calculate_dice(pred, gt, threshold=0.5):
    """Calculate Dice Coefficient"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
    
    return dice

def calculate_precision_recall_f1(pred, gt, threshold=0.5):
    """Calculate Precision, Recall, and F1-Score"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)
    
    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))
    fn = np.sum((1 - pred_binary) * gt_binary)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1

def calculate_mae(pred, gt):
    """Calculate Mean Absolute Error"""
    mae = np.mean(np.abs(pred - gt))
    return mae

def calculate_accuracy(pred, gt):
    """Calculate pixel-wise accuracy"""
    correct = np.sum(pred == gt)
    total = pred.size
    accuracy = correct / total
    return accuracy

def evaluate_segmentation(pred_mask, gt_mask, threshold=0.5):
    """Comprehensive segmentation evaluation"""
    iou = calculate_iou(pred_mask, gt_mask, threshold)
    dice = calculate_dice(pred_mask, gt_mask, threshold)
    precision, recall, f1 = calculate_precision_recall_f1(pred_mask, gt_mask, threshold)
    mae = calculate_mae(pred_mask, gt_mask)
    accuracy = calculate_accuracy((pred_mask > threshold).astype(int), 
                                  (gt_mask > threshold).astype(int))
    
    metrics = {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae,
        'accuracy': accuracy
    }
    
    return metrics

def batch_evaluate(pred_masks, gt_masks, threshold=0.5):
    """Evaluate batch of masks"""
    all_metrics = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        metrics = evaluate_segmentation(pred, gt, threshold)
        all_metrics.append(metrics)
    
    # Average
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics