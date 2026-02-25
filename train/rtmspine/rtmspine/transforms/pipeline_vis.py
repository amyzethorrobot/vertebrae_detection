from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
import matplotlib.pyplot as plt

@TRANSFORMS.register_module()
class PipelineVisualizer(BaseTransform):
    """

    Required Keys:

        - img
        - keypoints

    Args:
        indents (Tuple[int, int]): Indents from axis borders
    """

    def __init__(self, 
                 indents: Union[Tuple[int, int], int] = 0) -> None:
        super().__init__()

        if isinstance(indents, Tuple):
            self.indent_x = indents[0]
            self.indent_y = indents[1]
        elif isinstance(indents, int):
            self.indent_x = indents
            self.indent_y = indents
        else:
            raise ValueError('indents must be list or int')


    def transform(self, results: Dict):
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict
        """

        keypoints_x, keypoints_y = results['transformed_keypoints'].transpose()
        img_path = results['img_path']

        min_x = 0
        max_x = results['input_size'][0]
        min_y = 0
        max_y = results['input_size'][1]
        indent_x = self.indent_x
        indent_y = self.indent_y

        keypoints_in_bbox = (np.where(keypoints_x > min_x, 1, 0)
                             * np.where(keypoints_x < max_x, 1, 0)
                             * np.where(keypoints_y > min_y, 1, 0)
                             * np.where(keypoints_y < max_y, 1, 0)).astype(bool)
        

        plt.figure(figsize=(5, 10))
        plt.imshow(results['img'])
        plt.scatter(keypoints_x[keypoints_in_bbox], keypoints_y[keypoints_in_bbox], color = "red", s = 10)
        plt.xlim(min_x  - indent_x, max_x + indent_x)
        plt.ylim(max_y + indent_y, min_y - indent_y)
        plt.show()

        return results