"""
Common metric calculation functions for image segmentation tasks.
All functions expect binary masks (0 or 1) as NumPy arrays.
"""
import numpy as np

def accuracy(groundtruth_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculates pixel accuracy between two binary masks.

    Args:
        groundtruth_mask (np.ndarray): The ground truth binary mask.
        pred_mask (np.ndarray): The predicted binary mask.

    Returns:
        float: The accuracy score, rounded to 3 decimal places.
               Returns 1.0 if both masks are empty and correctly predicted,
               or 0.0 if division by zero occurs due to other empty mask mismatches.
    """
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == pred_mask) # Correctly predicted pixels (TP + TN)
    
    denominator = union + xor - intersect 
    if denominator == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(groundtruth_mask) == 0 else 0.0
    
    acc = np.mean(xor / denominator) 
    return round(acc, 3)

def dice_coefficient(groundtruth_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculates the Dice coefficient (F1 score) between two binary masks.

    Args:
        groundtruth_mask (np.ndarray): The ground truth binary mask.
        pred_mask (np.ndarray): The predicted binary mask.

    Returns:
        float: The Dice coefficient, rounded to 3 decimal places.
               Returns 1.0 if both masks are empty (perfect agreement).
    """
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    if total_sum == 0:
        return 1.0 if intersect == 0 else 0.0 
    dice = np.mean(2 * intersect / total_sum)
    return round(dice, 3)

def iou_score(groundtruth_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) or Jaccard index.

    Args:
        groundtruth_mask (np.ndarray): The ground truth binary mask.
        pred_mask (np.ndarray): The predicted binary mask.

    Returns:
        float: The IoU score, rounded to 3 decimal places.
               Returns 1.0 if both masks are empty.
    """
    intersection = np.logical_and(groundtruth_mask, pred_mask)
    union = np.logical_or(groundtruth_mask, pred_mask)
    sum_intersection = np.sum(intersection)
    sum_union = np.sum(union)
    if sum_union == 0:
        return 1.0 if sum_intersection == 0 else 0.0
    iou = sum_intersection / sum_union
    return round(iou, 3)
