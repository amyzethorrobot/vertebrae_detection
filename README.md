# Vertebrae Detection

## Description

This repository contains code for MICCAI conference paper. It allows user to train his own model using our pipeline and then test it on our open test dataset.

The repository is organised as follows:

- `train/rtmspine` - contains custom models and scripts for training
  - `train.py` - script for training the model
  - `test.py` - script for testing the model
  - `configs/` - folder with model configuration files which can be used for training (additional info about configs for MMPose models at: <https://mmpose.readthedocs.io/en/latest/user_guides/configs.html>)
  - `rtmspine/` - folder with all custom modules and extra code
- `test_dataset/` - folder with all files for our open dataset
  - `test_images/` - folder with all test images
  - `test_annot.json` - annotation file for them
- `visualizer.ipynb` - notebook designed to visualise the result of tests of the model on our open dataset

## Requierments

```requirements
torch == 2.4.1
albumentations == 2.0.7
mmengine == 0.10.5
mmpose == 1.3.2
```

Note that the repository was tested with these specific versions of dependencies. MMPose installation instructions can be found at:
<https://mmpose.readthedocs.io/en/latest/installation.html>

## Training

Train model with script `train\rtmspine\train.py`. This script uses the same arguments and based on same principles as training script from MMPose framework. Manual for manipulating training parameters and managing training process of an mmpose model can be found at: <https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html>

## Testing

In the `test_dataset` folder you can find images and annotation file in `.json` format to evaluate the model.

Our codebase uses testing script `train\rtmspine\test.py` almost identical to one in MMPose framework to compute metrics, save processed images and write the output of the model into dump file in `.pkl` format. Additional manual about testing an mmpose model with custom arguments can be found at: <https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html>

To proper visualise predicted and ground truth keypoints from dump file you can use `visualiser.ipynb`. It has its own manual which van help to set the testing process properly.
