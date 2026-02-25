import numpy as np
from typing import Tuple
from mmpose.evaluation.functional import keypoint_pck_accuracy
from .spine_eval import vert_pck_accuracy, calc_endplate_sizes
from .spine_measures import vert_calc_distances, vert_makenorm_distances, vert_sizes


def split_spine(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, segments: dict):

    '''
    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (D=2 only)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        dict of preds np.ndarray[N, K_seg, D]
        dict of gts np.ndarray[N, K_seg, D]
        dict masks np.ndarray[N, K_seg]
        where K_seg - numer of keypoints in each segment
    '''

    segments_names = segments.keys()

    segments_pred = {}
    segments_gt = {}
    segments_mask = {}

    for k in segments_names:

        first_index = segments[k][0]
        last_index = segments[k][1]

        segments_pred[k] = pred[:, first_index:last_index, :]
        segments_gt[k] = gt[:, first_index:last_index, :]
        segments_mask[k] = mask[:, first_index:last_index]

    return segments_pred, segments_gt, segments_mask


def spine_segment_bbox(segments_gt: dict, segments_mask: dict, individual: bool = False) -> dict:

    '''
    Args:

        segments_gt (dict in form {seg: np.ndarray[N, K_seg, D]}): \
            groundtruth keypoints location for each segment 
        segments_mask (dict in form {seg: np.ndarray[N, K_seg]}): \
            visibility of keypoints for each segment
        individual (bool): use individual x_size and y_size bbox sizes \
            (if false uses max(x_size, y_size))

    Returns:

        dict in form {seg: np.ndarray[N, 2]}: batched bbox sizes for each segment 
    '''

    segments_names = segments_gt.keys()
    segments_bbox_sizes = {}
    N, K, _ = segments_gt[list(segments_names)[0]].shape

    for k in segments_names:

        segments_bbox_sizes[k] = np.zeros((N, 2))

        for n, gt_keypoints in enumerate(segments_gt[k]):

            mask = segments_mask[k][n].astype(bool)

            if sum(mask) == 0:
                segments_bbox_sizes[k][n][0] = 0
                segments_bbox_sizes[k][n][1] = 0
                continue

            x_coords = gt_keypoints[mask][:, 0]
            y_coords = gt_keypoints[mask][:, 1]
            size_x = np.abs(np.amax(x_coords) - np.amin(x_coords))
            size_y = np.abs(np.amax(y_coords) - np.amin(y_coords))

            if individual:
                segments_bbox_sizes[k][n][0] = size_x
                segments_bbox_sizes[k][n][1] = size_y
            else:
                max_size = np.amax([size_x, size_y])
                segments_bbox_sizes[k][n][0] = max_size
                segments_bbox_sizes[k][n][1] = max_size
    
    return segments_bbox_sizes


def spine_bbox(gt: np.ndarray, mask: np.ndarray, individual: bool = False):

    N = gt.shape[0]
    bbox_sizes = np.zeros((N, 2))

    for n, gt_keypoints in enumerate(gt):

        _mask = mask[n].copy()
        _mask[-2:] = 0
        _mask = _mask.astype(bool)
        
        x_coords = gt_keypoints[_mask][:, 0]
        y_coords = gt_keypoints[_mask][:, 1]
        size_x = np.abs(np.amax(x_coords) - np.amin(x_coords))
        size_y = np.abs(np.amax(y_coords) - np.amin(y_coords))

        if individual:
            bbox_sizes[n][0] = size_x
            bbox_sizes[n][1] = size_y
        else:
            max_size = np.amax([size_x, size_y])
            bbox_sizes[n][0] = max_size
            bbox_sizes[n][1] = max_size

    return bbox_sizes


def pck_accuracy_segments(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, segments: dict, treshold: float) -> dict:

    segments_names = segments.keys()
    segments_pred, segments_gt, segments_mask = split_spine(pred, gt, mask, segments)
    #segments_bbox_sizes = spine_segment_bbox(segments_gt, segments_mask)

    segments_bbox_sizes = {k: spine_bbox(gt, mask) for k in segments_names}

    segments_pck = {}
    for k in segments_names:
        _, pck, _ = keypoint_pck_accuracy(segments_pred[k], 
                                          segments_gt[k], 
                                          segments_mask[k].astype(bool), 
                                          treshold, 
                                          segments_bbox_sizes[k])

        segments_pck[k] = pck

    return segments_pck


def pckvert_segments(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, segments: dict, treshold: float, radius: str = 'global') -> dict:

    segments_names = segments.keys()
    segments_pred, segments_gt, segments_mask = split_spine(pred, gt, mask, segments)
    #segments_bbox_sizes = spine_segment_bbox(segments_gt, segments_mask)

    segments_bbox_sizes = {k: spine_bbox(gt, mask) for k in segments_names}

    segments_pck = {}
    for k in segments_names:

        pck = 0
        N = len(segments_pred[k])
        for n in range(N):

            endplates_size = calc_endplate_sizes(segments_gt['full'][n], segments_mask['full'][n])
            norm_distances = np.mean(endplates_size[endplates_size > 0])

            _pck = vert_pck_accuracy(segments_pred[k][n],
                                     segments_gt[k][n],
                                     segments_mask[k][n],
                                     treshold,
                                     radius = radius,
                                     exclude = None,
                                     global_fixed_norm = norm_distances)
            
            if _pck > 0:
                pck += _pck

        segments_pck[k] = pck/N

    return segments_pck


def distance_segments(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, segments: dict, error_treshold: float) -> Tuple:

    segments_names = segments.keys()
    segments_pred, segments_gt, segments_mask = split_spine(pred, gt, mask, segments)
    segments_bbox_sizes = {k: spine_bbox(gt, mask) for k in segments_names}

    segments_abs_distance = {}
    segments_rel_local = {}
    segments_rel_global = {}

    for k in segments_names:

        abs_distance = vert_calc_distances(segments_pred[k],
                                           segments_gt[k],
                                           segments_mask[k].astype(bool))
        
        bbox_size = segments_bbox_sizes['full']
        max_error = bbox_size[:, :1] * error_treshold

        N, K, _ = segments_pred[k].shape
        endplates = np.zeros((N, K//2))

        for n in range(0, len(abs_distance)):
            abs_distance[n][abs_distance[n] > max_error[n]] = max_error[n]
            endplates[n] = calc_endplate_sizes(segments_gt[k][n], segments_mask[k][n].astype(bool))
        
        segments_abs_distance[k] = np.mean(abs_distance[abs_distance > 0])
        
        rel_local = vert_makenorm_distances(abs_distance, endplates)

        segments_rel_local[k] = np.mean(rel_local[rel_local > 0])

        bboxes = np.ones((N, K//2)) * bbox_size[:, :1]
        rel_global = vert_makenorm_distances(abs_distance, bboxes)
        segments_rel_global[k] = np.mean(rel_global[rel_global > 0])
        
    return segments_abs_distance, segments_rel_local, segments_rel_global