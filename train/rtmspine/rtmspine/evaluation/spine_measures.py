from typing import Dict, Optional, Sequence, Union
import numpy as np

def vert_calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    
    '''Calculate distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        np.ndarray[N, K]: Distances. \
            If target keypoints are missing, the distance is -1.
    '''
    
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    distances[_mask] = np.linalg.norm((preds - gts)[_mask], axis=-1)
    
    return distances


def vert_sizes(gt_keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:

    '''Calculate sizes of endplates of vertebrae 
    from gt annotations for 1 sample (not batched!).

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        gt_keypoints (np.ndarray[1, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[1, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        np.ndarray[K//2]: vertebrae sizes.
            If target keypoints are missing, the size is -1.
    '''

    _, keypoint_num, _ = gt_keypoints.shape

    _mask = mask.copy()[0]
    # endplates [K//2, 2]
    endplates = np.array([[i, i+1] for i in range(0, keypoint_num, 2)])
    ep_visible = np.array([_mask[i] * _mask[i+1] for i in range(0, keypoint_num, 2)]).astype(bool)
    ep_num = len(endplates)
    # endplate_sizes [K//2]
    endplate_sizes = np.full(ep_num, -1)

    for i, endplate in enumerate(endplates):

        if not ep_visible[i]:
            continue

        first = endplate[0]
        second = endplate[1]
        endplate_sizes[i] = np.linalg.norm(gt_keypoints[0][first] 
                                           - gt_keypoints[0][second])

    return endplate_sizes


def vert_makenorm_distances(abs_disctances: np.ndarray, norm_factors: np.ndarray) -> np.ndarray:

    '''Calculates distances relative to norm_factors

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        abs_distances (np.ndarray[N, K]): raw distances 
        between preds and targets in batch
        norm_factors (np.ndarray[N, D]): normalization factor.
            relative_distance = abs_distance/norm_factor

    Returns:
        np.ndarray[K]: normalized distances.
            If target keypoints are missing, the distance is -1.
    
    '''

    # distances: [N, K]
    normalized_distances = abs_disctances.copy()
    N, _ = abs_disctances.shape

    _norm_factors = norm_factors.repeat(2, axis = 1)

    for n in range(N):

        _mask = np.where(_norm_factors[n] > 0, 1, 0).astype(bool)
        normalized_distances[n][_mask] /= _norm_factors[n][_mask]

    return normalized_distances

# def vert_relative_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray) -> np.ndarray:

#     endplate_sizes = vert_sizes(gts, mask)
#     distances = vert_calc_distances(preds, gts, mask)

#     for i, endplate_size in enumerate(endplate_sizes):

#         if endplate_size <= 0:
#             continue

#         distances[i * 2] /= endplate_size
#         distances[i * 2 + 1] /= endplate_size

#     return distances
