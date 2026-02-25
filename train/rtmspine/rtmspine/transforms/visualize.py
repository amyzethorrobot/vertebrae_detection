import os
import numpy as np
from typing import Optional, Dict, Any
from mmcv import BaseTransform
from mmpose.visualization import PoseLocalVisualizer
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from mmpose.datasets.datasets.utils import parse_pose_metainfo

from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class VisualizeTransform(BaseTransform):
    """
    Custom transform for visualization in MMPose pipelines using PoseLocalVisualizer.

    This transform visualizes keypoints and bboxes using MMPose's PoseLocalVisualizer,
    supporting both saving to disk and interactive display, fully supporting usage
    with PoseDataSample and InstanceData, and allows explicit dataset_meta specification.

    Args:
        save_dir (str, optional): Directory to save visualizations if mode='save'.
        mode (str): Visualization mode, either 'save' or 'show'.
        draw_bbox (bool): Draw bounding boxes on the image.
        draw_kpt (bool): Draw keypoints on the image.
        window_name (str, optional): Window name for display mode.
        dataset_meta (dict, optional): Metadata (e.g. skeleton_links) for visualization.
        radius (int): Keypoint circle radius for visualizer.
        **kwargs: Other PoseLocalVisualizer options.
    """
    def __init__(self,
                 save_dir: Optional[str] = 'vis_results',
                 mode: str = 'show',
                 draw_bbox: bool = True,
                 draw_kpt: bool = True,
                 window_name: Optional[str] = 'Pose Visualization',
                 dataset_meta: Optional[Dict[str, Any]] = None,
                 radius: int = 3,
                 **kwargs):
        super().__init__()
        assert mode in ['save', 'show'], "mode must be either 'save' or 'show'"
        self.save_dir = save_dir
        self.mode = mode
        self.draw_bbox = draw_bbox
        self.draw_kpt = draw_kpt
        self.window_name = window_name or 'Pose Visualization'

        if isinstance(dataset_meta, dict) and 'from_file' in dataset_meta:
            dataset_meta = parse_pose_metainfo(dataset_meta)

        self.dataset_meta = dataset_meta
        self.radius = radius
        save_dir = None
        if self.mode == 'save':
            os.makedirs(self.save_dir, exist_ok=True)
            save_dir = self.save_dir


        # Visualizer is initialized with radius and any other kwargs
        self.visualizer = PoseLocalVisualizer(save_dir=save_dir, radius=self.radius, **kwargs)
        if self.dataset_meta is not None:
            self.visualizer.set_dataset_meta(self.dataset_meta)


    def transform(self, results):
        """
        Visualize current sample as a PoseDataSample/InstanceData.

        Supports display (blocks until key press) or saving to disk.
        """
        # Retrieve the image in [H, W, C] np.uint8 format
        img = results['img'] if 'img' in results else results['inputs']
        img_path = results.get('img_path', None)
        out_file = None
        if self.mode == 'save':
            fname = os.path.basename(img_path) if img_path else 'vis.jpg'
            out_file = os.path.join(self.save_dir, fname)

        # Convert to numpy if needed
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if img.max() <= 1.5:
            img = (img * 255).astype(np.uint8)
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # --- Convert keypoints/bboxes to PoseDataSample if not already present ---
        pose_sample = results.get('data_samples', None)
        if pose_sample is None:
            # Try to create from keypoints/bboxes fields
            instance = InstanceData()
            instance.keypoints = results.get('keypoints', None)
            if 'transformed_keypoints' in results:
                instance.transformed_keypoints = results['transformed_keypoints']

            instance.keypoints_visible = results.get('keypoints_visible', None)
            instance.bboxes = results.get('bbox', None)

            pose_sample = PoseDataSample()
            # Attach as gt_instances if ground truth, or pred_instances otherwise
            pose_sample.gt_instances = instance

        if 'id' in results:
            print('Visualize id ', results['id'])
        # Now visualize
        self.visualizer.add_datasample(
            name=self.window_name,
            image=img,
            data_sample=pose_sample,
            draw_bbox=self.draw_bbox,
            show_kpt_idx=self.draw_kpt,
            out_file=out_file if self.mode == 'save' else None,
            show=self.mode == 'show',
            wait_time=0 if self.mode == 'save' else 0
        )

        # If displaying, make sure block until key press
        if self.mode == 'show':
            # If matplotlib is used inside visualizer, this ensures blocking
            try:
                import matplotlib.pyplot as plt
                plt.show(block=True)
            except Exception:
                pass

        return results

