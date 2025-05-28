import numpy as np

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == pred_mask)
    # Avoid division by zero if union + xor - intersect is zero
    denominator = union + xor - intersect
    if denominator == 0:
        # Return 1.0 if both masks are empty and correctly predicted as empty,
        # or handle as per specific requirements (e.g., return 0 or NaN).
        # Assuming 1.0 for perfect match on empty areas.
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(groundtruth_mask) == 0 else 0.0
    acc = np.mean(xor / denominator)
    return round(acc, 3)

def dice_coefficient(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    if total_sum == 0:
        # If both masks are empty, dice is 1 (perfect agreement)
        return 1.0 if intersect == 0 else 0.0 
    dice = np.mean(2 * intersect / total_sum)
    return round(dice, 3)

def iou_score(groundtruth_mask, pred_mask):
    intersection = np.logical_and(groundtruth_mask, pred_mask)
    union = np.logical_or(groundtruth_mask, pred_mask)
    sum_intersection = np.sum(intersection)
    sum_union = np.sum(union)
    if sum_union == 0:
        # If union is 0, it means both masks are empty. IoU is 1.
        return 1.0 if sum_intersection == 0 else 0.0
    iou = sum_intersection / sum_union
    return round(iou, 3)
