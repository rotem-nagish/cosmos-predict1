# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of dataset settings and augmentations for tokenization

Run this command to interactively debug:
PYTHONPATH=$(pwd) python -m \
cosmos_predict1.tokenizer.training.datasets.dataset_provider \
    --dataset_name hdvila_video \
    --is_train
"""
import os

from torch.utils.data._utils.collate import default_collate

from cosmos_predict1.tokenizer.training.datasets.augmentation_provider import (
    video_train_augmentations,
    video_val_augmentations,
)
from cosmos_predict1.tokenizer.training.datasets.utils import categorize_aspect_and_store
from cosmos_predict1.tokenizer.training.datasets.video_dataset import Dataset
from cosmos_predict1.utils.lazy_config import instantiate

BASE_DIR = "/mnt/rylo-tnas/users/rotem/sign/data" if "/mnt/rylo-tnas" in os.getcwd() else "/workspace/datasets"


def pose_collate_fn(batch):
    """Collate that keeps gt_pose Pose objects as a list instead of stacking."""
    if not batch:
        return {}
    result = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        if key == "gt_pose":
            result[key] = values
        else:
            try:
                result[key] = default_collate(values)
            except Exception:
                result[key] = values
    return result

_VIDEO_PATTERN_DICT = {
    "hdvila_video": "datasets/hdvila/videos/*.mp4",
    "sl_ft_video": "/home/rotem/sign/data/*/*.mp4"
}

# Grid dataset: 128x512 grid videos -> original dictio*.mp4 reconstruction targets
_GRID_VIDEO_PATTERN = f"{BASE_DIR}/128x512_grid_videos/dictio*.mp4"
_GRID_ORIGINAL_DIR = f"{BASE_DIR}/processed/sign-tube/videos"


def apply_augmentations(data_dict, augmentations_dict):
    """
    Loop over each LazyCall object and apply it to data_dict in place.
    """
    for aug_name, lazy_aug in augmentations_dict.items():
        aug_instance = instantiate(lazy_aug)
        data_dict = aug_instance(data_dict)
    return data_dict


class AugmentDataset(Dataset):
    def __init__(self, base_dataset, augmentations_dict):
        """
        base_dataset: the video dataset instance
        augmentations_dict: the dictionary returned by
                            video_train_augmentations() or video_val_augmentations()
        """
        self.base_dataset = base_dataset

        # Pre-instantiate every augmentation ONCE:
        self.augmentations = []
        for aug_name, lazy_aug in augmentations_dict.items():
            aug_instance = instantiate(lazy_aug)  # build the actual augmentation
            self.augmentations.append((aug_name, aug_instance))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get the raw sample from the base dataset
        data = self.base_dataset[index]
        data = categorize_aspect_and_store(data)

        # Apply each pre-instantiated augmentation
        for aug_name, aug_instance in self.augmentations:
            data = aug_instance(data)

        return data


def dataset_entry(
    dataset_name: str,
    dataset_type: str,
    is_train: bool = True,
    resolution="720",
    crop_height=256,
    num_video_frames=25,
    pose_normalization='mean_std',
    limit=None,
    mask_type='bb',
) -> AugmentDataset:
    if dataset_type != "video":
        raise ValueError(f"Dataset type {dataset_type} is not supported")

    # Instantiate the video dataset
    base_dataset = Dataset(
        video_pattern=_VIDEO_PATTERN_DICT[dataset_name.lower()],
        num_video_frames=num_video_frames,
        pose_normalization=pose_normalization,
        limit=limit,
        mask_type=mask_type,
    )

    # Pick the training or validation augmentations
    if is_train:
        aug_dict = video_train_augmentations(
            input_keys=["video"],  # adjust if necessary
            resolution=resolution,
            crop_height=crop_height,
        )
    else:
        aug_dict = video_val_augmentations(
            input_keys=["video"],
            resolution=resolution,
            crop_height=crop_height,
        )

    # Wrap the dataset with the augmentations
    return AugmentDataset(base_dataset, aug_dict)


def grid_dataset_entry(
    is_train: bool = True,
    num_video_frames: int = 49,
    pose_normalization: str = "minmax",
    limit=None,
    target_size: tuple = (256, 256),
    crop_height: int = 256,  # unused -- kept for interface consistency with dataset_entry
    part_fusion: bool = False,
    fliplr_list_path: str = None,
    mask_type: str = 'bb',
) -> AugmentDataset:
    """Return a dataset that encodes 256x256 grid videos and reconstructs the original."""
    from cosmos_predict1.tokenizer.training.datasets.grid_video_dataset import GridVideoDataset
    from cosmos_predict1.tokenizer.training.datasets.augmentation_provider import (
        grid_video_train_augmentations,
        grid_video_val_augmentations,
    )

    base_dataset = GridVideoDataset(
        grid_video_pattern=_GRID_VIDEO_PATTERN,
        original_video_dir=_GRID_ORIGINAL_DIR,
        num_video_frames=num_video_frames,
        pose_normalization=pose_normalization,
        limit=limit,
        target_size=target_size,
        part_fusion=part_fusion,
        fliplr_list_path=fliplr_list_path,
        mask_type=mask_type,
    )

    aug_dict = (
        grid_video_train_augmentations()
        if is_train
        else grid_video_val_augmentations()
    )
    return AugmentDataset(base_dataset, aug_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="SL_ft")
    parser.add_argument("--dataset_type", default="video")
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--crop_height", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=25)
    args = parser.parse_args()

    dataset = dataset_entry(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        is_train=args.is_train,
        resolution=args.resolution,
        crop_height=args.crop_height,
        num_video_frames=args.num_frames,
    )

    print(f"Total samples: {len(dataset)}")

    # 3) Grab one sample (or a few) to check shapes, keys, etc.
    if len(dataset) > 0:
        sample_idx = 0
        sample = dataset[sample_idx]
        print(f"Sample index {sample_idx} keys: {list(sample.keys())}")
        if "video" in sample:
            print("Video shape:", sample["video"].shape)
        if "video_name" in sample:
            print("Video metadata:", sample["video_name"])
        print("---\nSample loaded successfully.\n")
    else:
        print("Dataset has no samples!")
