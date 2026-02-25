import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.functional import (keypoint_auc, keypoint_epe, keypoint_nme,
                          keypoint_pck_accuracy)

from .spine_eval import (vert_pck_accuracy, vert_relative_distances, vert_pck_all, vert_pck_all_segments)
from .spine_measures import (vert_calc_distances, vert_sizes, vert_makenorm_distances)


@METRICS.register_module()
class PCKVert(BaseMetric):
    """PCK accuracy evaluation metric.
    Calculate the pose accuracy of Percentage of Correct Keypoints (PCK) for
    each individual keypoint and the averaged accuracy across all keypoints.
    PCK metric measures accuracy of the localization of the body joints.
    The distances between predicted positions and the ground-truth ones
    are typically normalized by the person bounding box size.
    The threshold (thr) of the normalized distance is commonly set
    as 0.05, 0.1 or 0.2 etc.
    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)
    Args:
        thr(float): Threshold of PCK calculation. Default: 0.05.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        exclude_keypoints (list, optional): Indices of keypoints to not include in pck metric.
        batch_pck (bool): Compute pck for all keypoints in all samples instead of averaging pck of individual samples
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    def __init__(self,
                 thr: float = 0.05,
                 collect_device: str = 'cpu',
                 exclude_keypoints: Optional[list] = None,
                 batch_pck: bool = False,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.thr = thr
        self.exclude_keypoints = exclude_keypoints
        self.batch_pck = batch_pck

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metrics = dict()

        logger.info(f'Evaluating {self.__class__.__name__} '
                    f'(normalized by ``"bbox_size"``)...')

        if self.batch_pck:

            pck = vert_pck_all(pred_coords, gt_coords, mask, self.thr, radius='global', exclude = self.exclude_keypoints)

        else:
        
            N = len(pred_coords)
            pck = 0
            for n in range(N):
                pck += vert_pck_accuracy(pred_coords[n], gt_coords[n], mask[n], self.thr, radius='global', exclude = self.exclude_keypoints)
            pck = pck/N

        metrics['PCKVert'] = pck

        return metrics
    

@METRICS.register_module()
class DistanceVert(BaseMetric):
    """
    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)
    Args:
        thr(float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso', which correspond
            to 'PCK', 'PCKh' and 'tPCK' respectively. Default: ``'bbox'``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            vert_sizes_array = vert_sizes(gt['keypoints'], mask).reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'vert_sizes_array': vert_sizes_array
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])
        # vert_sizes_array: [N, K//2]
        vert_sizes_array = np.concatenate([result['vert_sizes_array'] for result in results])

        metrics = dict()
        
        absolute_distances = vert_calc_distances(pred_coords, gt_coords, mask)

        #relative_distances = vert_relative_distances(pred_coords, gt_coords, mask)
        relative_distances = vert_makenorm_distances(absolute_distances, vert_sizes_array)

        metrics['DistanceVert_ABS'] = np.mean(absolute_distances[absolute_distances > 0])
        metrics['DistanceVert_REL'] = np.mean(relative_distances[relative_distances > 0])

        return metrics