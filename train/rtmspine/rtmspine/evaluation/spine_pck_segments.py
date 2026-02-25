from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS

from .spine_functions import pck_accuracy_segments, pckvert_segments, distance_segments, spine_bbox


@METRICS.register_module()
class PCKSegments(BaseMetric):
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
                 segments: dict,
                 thr: float = 0.05,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.thr = thr
        self.segments = segments

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

        pck = pck_accuracy_segments(pred_coords, gt_coords, mask, self.segments, self.thr)

        for k in pck.keys():

            key = 'PCKSegments_' + k
            metrics[key] = pck[k]

        return metrics
    


@METRICS.register_module()
class PCKVertSegments(BaseMetric):
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
                 segments: dict,
                 thr: float = 0.05,
                 radius: str = 'global',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.thr = thr
        self.segments = segments
        self.radius = radius

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

        pck = pckvert_segments(pred_coords, gt_coords, mask, self.segments, self.thr, radius = self.radius)

        for k in pck.keys():

            key = 'PCKVertSegments/' + k
            metrics[key] = pck[k]

        return metrics
    

@METRICS.register_module()
class DistanceSegments(BaseMetric):
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
                 segments: dict,
                 error_treshold: float = 0.05,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.error_treshold = error_treshold
        self.segments = segments

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

        distance_abs, distance_local, distance_global = distance_segments(pred_coords, gt_coords, mask, self.segments, self.error_treshold)

        for k in distance_abs.keys():

            key = 'DistanceABS/' + k
            metrics[key] = distance_abs[k]

            key = 'DistanceREL-L/' + k
            metrics[key] = distance_local[k]

            key = 'DistanceREL-G/' + k
            metrics[key] = distance_global[k]

        metrics['MEAN BBOX'] = np.mean(spine_bbox(gt_coords, mask))

        return metrics