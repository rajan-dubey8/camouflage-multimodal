"""
Utility Functions for Multimodal Camouflage Detection
"""

from .metrics import (
    calculate_iou,
    calculate_dice,
    calculate_precision_recall_f1,
    calculate_mae,
    evaluate_segmentation,
    batch_evaluate
)

from .visualization import (
    plot_training_history,
    plot_attention_heatmap,
    plot_comparison,
    plot_metrics_summary
)

__all__ = [
    'calculate_iou',
    'calculate_dice',
    'calculate_precision_recall_f1',
    'calculate_mae',
    'evaluate_segmentation',
    'batch_evaluate',
    'plot_training_history',
    'plot_attention_heatmap',
    'plot_comparison',
    'plot_metrics_summary'
]