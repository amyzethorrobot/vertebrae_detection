from itertools import product
from typing import Optional, Tuple, Union, List

import numpy as np

from mmpose.codecs.utils import get_simcc_maximum
from mmpose.codecs.utils.refinement import refine_simcc_dark
from mmpose.registry import KEYPOINT_CODECS
from mmpose.codecs.simcc_label import SimCCLabel


@KEYPOINT_CODECS.register_module()
class SimCCFineInvisible(SimCCLabel):
    r"""Generate keypoint representation via "SimCC" approach.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
            The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wx=w*simcc_split_ratio`
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
            The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wy=h*simcc_split_ratio`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.
        use_dark (bool): Whether to use the DARK post processing. Defaults to
            False.
        decode_visibility (bool): Whether to decode the visibility. Defaults
            to False.
        decode_beta (float): The beta value for decoding visibility. Defaults
            to 150.0.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """


    def __init__(
            self,
            input_size: Tuple[int, int],
            smoothing_type: str = 'gaussian',
            sigma: Union[float, int, Tuple[float]] = 6.0,
            simcc_split_ratio: float = 2.0,
            label_smooth_weight: float = 0.0,
            normalize: bool = True,
            use_dark: bool = False,
            decode_visibility: bool = False,
            decode_beta: float = 150.0,

            no_fine: List[Union[int, Tuple[int, int], List[int]]] = []
    ) -> None:
        super().__init__(input_size,
                         smoothing_type,
                         sigma,
                         simcc_split_ratio,
                         label_smooth_weight,
                         normalize,
                         use_dark,
                         decode_visibility,
                         decode_beta)


        self.no_fine_mask = None
        self.no_fine_list = no_fine


    @staticmethod
    def build_boolean_mask(length: int, no_fine: List[Union[int, Tuple[int, int], List[int]]] = []) -> np.ndarray:
        """
        Create boolean numpy array of size 'length'.
        Set True at positions specified in no_fine (single indices or ranges).

        Args:
            length (int): Length of the output array.
            no_fine (List[Union[int, Tuple[int, int], List[int]]]): List of indices or ranges (start, end).

        Returns:
            np.ndarray: Boolean array of shape (length,)
        """
        mask = np.zeros(length, dtype=bool)

        for item in no_fine:
            # Case: single index
            if isinstance(item, int):
                if 0 <= item < length:
                    mask[item] = True

            # Case: range as tuple or list of two ints
            elif (isinstance(item, (tuple, list)) and len(item) == 2
                  and all(isinstance(x, int) for x in item)):
                start, end = item
                start = max(0, start)
                end = min(length, end)
                if start < end:
                    mask[start:end] = True

            else:
                raise ValueError(f"Invalid item in no_fine list: {item}")

        return mask

    def _map_coordinates(
            self,
            keypoints: np.ndarray,
            keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mapping keypoint coordinates into SimCC space."""

        # 1) Check if invisible
        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)

        keypoint_weights = np.ones(keypoints_visible.shape)
        skipping_mask = np.logical_and(np.abs(keypoints_visible) < 0.5, self.no_fine_mask[None, :])
        keypoint_weights[skipping_mask] = 0

        return keypoints_split, keypoint_weights

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints into SimCC labels. Note that the original
        keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_x_labels (np.ndarray): The generated SimCC label for
                x-axis.
                The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wx=w*simcc_split_ratio`
            - keypoint_y_labels (np.ndarray): The generated SimCC label for
                y-axis.
                The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wy=h*simcc_split_ratio`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        # Override keypoints_visible
        kp_visible = keypoints_visible.copy()

        if kp_visible is None:
            kp_visible = np.ones(keypoints.shape[:2], dtype=np.float32)
        # Make invisible if out of frame

        w, h = self.input_size
        x, y = keypoints[:, :, 0], keypoints[:, :, 1]
        invisible_mask = np.logical_or(np.logical_or(x < 0, x >= w), np.logical_or(y < 0, y >= h))

        # Out of frame marker only for visible before kps
        kp_visible[invisible_mask] *= -1

        #if np.any(kp_visible < -0.5):
        #    print('Invisible mask active')


        if self.no_fine_mask is None:
            self.no_fine_mask = self.build_boolean_mask(keypoints.shape[1], self.no_fine_list)


        return super(SimCCFineInvisible, self).encode(keypoints, kp_visible)


