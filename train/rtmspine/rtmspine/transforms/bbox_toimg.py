from typing import Dict, Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.codecs import *  # noqa: F401, F403
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BboxToImage(BaseTransform):
    """Changes Bbox size to the size of the whole image

    Required Keys:

        - img_shape
        - bbox
        - bbox_center (optional)
        - bbox_scale (optional)

    Modified Keys:

        - bbox
        - bbox_center (optional)
        - bbox_scale (optional)
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BboxToImage`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        img_shape = results['img_shape']
        x0 = 0
        y0 = 0
        x1 = img_shape[1]
        y1 = img_shape[0]

        results['bbox'] = np.array([[x0, y0, x1, y1]])

        if results.get('bbox_center', None) is not None:

            bbox_center_x = (results['bbox'][0][0] + results['bbox'][0][2]) / 2
            bbox_center_y = (results['bbox'][0][1] + results['bbox'][0][3]) / 2
            results['bbox_center'] = np.array([[bbox_center_x, bbox_center_y]])

        if results.get('bbox_scale', None) is not None:
            bbox_scale_x = abs(results['bbox'][0][2] - results['bbox'][0][0])
            bbox_scale_y = abs(results['bbox'][0][3] + results['bbox'][0][1])
            results['bbox_scale'] = np.array([[bbox_scale_x, bbox_scale_y]])

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'()'
        return repr_str