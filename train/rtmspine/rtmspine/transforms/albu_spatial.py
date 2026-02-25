import warnings
from copy import deepcopy
from typing import List, Optional

import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness
from mmengine.dist import get_dist_info
from scipy.stats import truncnorm

from mmpose.codecs import *  # noqa: F401, F403
from mmpose.registry import TRANSFORMS
import matplotlib.pyplot as plt

try:
    import albumentations
except ImportError:
    albumentations = None

@TRANSFORMS.register_module()
@avoid_cache_randomness
class AlbumentationSL(BaseTransform):
    """Albumentation augmentation 
    add-on for spatial-level transforms

    This modification of basic Albumentation wrapper 
    adds support for spatial-level transforms which affect 
    keypoints and bboxes of data sample (ElasticTransform, RandomCrop, etc.)
    with functionality of original Albumentation preserved.

    Required Keys:

    - img

    Optinal supported keys:

    - keypoints
    - bbox

    Modified Keys:

    - img
    - keypoints (optional)
    - bbox (optional)
    - img_shape (optinal)

    Args:
        transforms (List[dict]): A list of Albumentation transforms.
            An example of ``transforms`` is as followed:
            .. code-block:: python

                [
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                ]
        keymap (dict | None): key mapping from ``input key`` to
            ``albumentation-style key``.
            Defaults to None, which will use {'img': 'image'}.
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms], 
            keypoint_params=dict(format='xy', 
                                 remove_invisible = False, 
                                 check_each_transform = False
                                 ),
            bbox_params=albumentations.BboxParams(format="pascal_voc"))

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            albumentations.BasicTransform: The constructed transform object
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            rank, _ = get_dist_info()
            if rank == 0 and not hasattr(
                    albumentations.augmentations.transforms, obj_type):
                warnings.warn(
                    f'{obj_type} is not pixel-level transformations. '
                    'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`Albumentation` to apply
        albumentations transforms.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): Result dict from the data pipeline.

        Return:
            dict: updated result dict.
        """

        # map result dict to albumentations format
        results_albu = {}
        for k, v in self.keymap_to_albu.items():
            assert k in results, \
                f'The `{k}` is required to perform albumentations transforms'
            results_albu[v] = results[k]

        if self.keymap_to_albu.get('keypoints', None) is not None:
            # notice for mmpose 1.3.2!
            # keypoints in mmpose results dict have shape [1, K, 2]
            # keypoints in albumentations are expected to have shape [K, 2]
            results_albu['keypoints'] = results['keypoints'][0]

        if self.keymap_to_albu.get('bbox', None) is not None:
            # notice for mmpose 1.3.2!
            # bbox in mmpose are stored as np.ndarray [[x0, y0, x1, y1]]
            # bboxes in albumentatins are stored as list [[x0, y0, x1, y1, category_id]]
            results_albu['bboxes'] = [list(results['bbox'][0])]
            results_albu['bboxes'][0].append(results['category_id'])
    
        # Apply albumentations transforms
        results_albu = self.aug(**results_albu)

        # Rewriting altered fields back to mmpose format if needed

        if self.keymap_to_albu.get('keypoints', None) is not None:
            results_albu['keypoints'] = np.array([results_albu['keypoints']])

        if self.keymap_to_albu.get('bbox', None) is not None:

            results_albu['bboxes'][0].pop(-1)

            # bbox_center and bbox_scale values are not affected 
            # by bbox-related albumentations transforms 
            # so they are recalculated manually

            bbox_center_x = (results_albu['bboxes'][0][0] + results_albu['bboxes'][0][2]) / 2
            bbox_center_y = (results_albu['bboxes'][0][1] + results_albu['bboxes'][0][3]) / 2
            results['bbox_center'] = np.array([[bbox_center_x, bbox_center_y]])

            bbox_scale_x = abs(results_albu['bboxes'][0][2] - results_albu['bboxes'][0][0])
            bbox_scale_y = abs(results_albu['bboxes'][0][3] + results_albu['bboxes'][0][1])
            results['bbox_scale'] = np.array([[bbox_scale_x, bbox_scale_y]])

        # map the albu results back to the original format
        for k, v in self.keymap_to_albu.items():
            results[k] = results_albu[v]

        # edit img_shape field if needed
        results['img_shape'] = np.shape(results['img'])[:-1]

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str