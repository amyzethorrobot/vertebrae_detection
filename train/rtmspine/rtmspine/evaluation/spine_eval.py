import numpy as np
from typing import Union

def calc_endplate_sizes(gt_keypoints: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''Calculate sizes of endplates
    from gt keypoints annotations for 1 sample (not batched!).

    Args:
        gt_keypoints (np.ndarray[K, D]): Groundtruth keypoint location.
        mask (np.ndarray[K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        np.ndarray[K//2]: vertebrae sizes.
            If target keypoints are missing, the size is -1.
    '''

    K, _ = gt_keypoints.shape

    # endplates [K//2, 2]
    endplates = np.array([[i, i+1] for i in range(0, K, 2)])
    endplates_visible = np.array([mask[i] * mask[i+1] for i in range(0, K, 2)]).astype(bool)
    # endplate_sizes [K//2]
    endplates_size = np.full(K//2, -1).astype(float)

    for i, plate in enumerate(endplates):
        if not endplates_visible[i]:
            continue

        first = plate[0]
        second = plate[1]
        endplates_size[i] = np.linalg.norm(gt_keypoints[first] - gt_keypoints[second])

    return endplates_size


def vert_pck_accuracy(pred_keypoints: np.ndarray,
                      gt_keypoints: np.ndarray,
                      mask: np.ndarray,
                      threshold: np.ndarray,
                      radius: str = 'global',
                      exclude: Union[list, None] = None,
                      global_fixed_norm: float = None) -> float:
    '''Computes pck metric with respect to size of vertebrae (average or individual)
    for 1 sample (not batched!)

    Args:
        pred_keypoints (np.ndarray[K, 2]): Predicted keypoint location.
        gt_keypoints (np.ndarray[K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        threshold (float): Threshold of PCK calculation.
        radius (str): radius type for pck metric.
            Can be:
            'global' - average endplate size is used for all keypoints
            'local' - corresponding endplate size is used for every keypoint
    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    '''

    mask_w_excluded = mask.copy()

    if exclude is not None:
        for i in exclude:
            mask_w_excluded[i] = 0

    visible_points = np.sum(mask_w_excluded)
    if visible_points < 1:
        return -1

    endplates_size = calc_endplate_sizes(gt_keypoints, mask)

    mask_w_excluded = mask_w_excluded.astype(bool)

    if radius == 'global':
        if global_fixed_norm is not None:
            norm_distances = global_fixed_norm
        else:
            norm_distances = np.mean(endplates_size[endplates_size > 0])
    elif radius == 'local':
        norm_distances = endplates_size.repeat(2)
    else:
        raise ValueError(f'Radius type must be \'global\' or \'local\', not \'{radius}\'')

    relative_distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis = 1) / norm_distances

    pck = np.sum(relative_distances[mask_w_excluded] < threshold) / visible_points

    return pck


def vert_relative_distances(pred_keypoints: np.ndarray,
                            gt_keypoints: np.ndarray,
                            mask: np.ndarray,
                            radius: str = 'global',
                            exclude: Union[list, None] = None) -> np.ndarray:
    
    '''Computes relative distances between prediction and ground truth 
    with respect to size of vertebrae (average or individual)
    for 1 sample (not batched!)

    Args:
        pred_keypoints (np.ndarray[K, 2]): Predicted keypoint location.
        gt_keypoints (np.ndarray[K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        radius (str): radius type for pck metric.
            Can be:
            'global' - average endplate size is used for all keypoints
            'local' - corresponding endplate size is used for every keypoint
    Returns:
        np.ndarray([K]): Relative distances for each keypoint. \
            For invisible and excluded points relative distance is equal to -1
    '''

    mask_w_excluded = mask.copy()

    if exclude is not None:
        for i in exclude:
            mask_w_excluded[i] = 0

    visible_points = np.sum(mask_w_excluded)
    if visible_points < 1:
        return -1

    endplates_size = calc_endplate_sizes(gt_keypoints, mask)

    mask_w_excluded = mask_w_excluded.astype(bool)

    if radius == 'global':
        norm_distances = np.mean(endplates_size[endplates_size > 0])
    elif radius == 'local':
        norm_distances = endplates_size.repeat(2)
    else:
        raise ValueError(f'Radius type must be \'global\' or \'local\', not \'{radius}\'')
    
    K, _ = pred_keypoints.shape
    relative_distances = np.full(K, -1).astype(float)

    relative_distances_raw = np.linalg.norm(pred_keypoints - gt_keypoints, axis = 1) / norm_distances
    relative_distances[mask_w_excluded] = relative_distances_raw[mask_w_excluded]

    return relative_distances

def vert_pck_all(pred_keypoints: np.ndarray,
                 gt_keypoints: np.ndarray,
                 mask: np.ndarray,
                 treshold: float,
                 radius: str = 'global',
                 exclude: Union[list, None] = None) -> float:
    
    N, K, _ = pred_keypoints.shape

    total_visible_points = 0
    points_above_treshold = 0

    for n in range(N):

        relative_distances_raw = vert_relative_distances(pred_keypoints[n],
                                                         gt_keypoints[n], 
                                                         mask[n], 
                                                         treshold = treshold, 
                                                         radius=radius, 
                                                         exclude = exclude)
        
        relative_distances = relative_distances_raw[relative_distances_raw > 0]
        
        total_visible_points += len(relative_distances)
        points_above_treshold += np.sum(relative_distances < treshold)

    pck = points_above_treshold / total_visible_points

    return pck


def vert_pck_all_segments(pred_keypoints: np.ndarray,
                 gt_keypoints: np.ndarray,
                 mask: np.ndarray,
                 treshold: float,
                 segments: dict,
                 radius: str = 'global',
                 exclude: Union[list, None] = None
                 ) -> float:
    
    N, K, _ = pred_keypoints.shape

    segments_names = segments.keys()
    pck_segments = {k: 0 for k in segments_names}
    visible_segments = {k: 0 for k in segments_names}

    for n in range(N):

        relative_distances_raw = vert_relative_distances(pred_keypoints[n],
                                                         gt_keypoints[n], 
                                                         mask[n], 
                                                         treshold = treshold, 
                                                         radius=radius, 
                                                         exclude = exclude)
        
        for k in segments_names:

            relative_distances_segment = relative_distances_raw[segments[k][0]:segments[k][1]]
            relative_distances_segment = relative_distances_segment[relative_distances_segment > 0]
            pck_segments[k] += np.sum(relative_distances_segment < treshold)
            visible_segments[k] += len(relative_distances_segment)
        
    for k in segments_names:
        pck_segments[k] /= visible_segments[k]

    return pck_segments