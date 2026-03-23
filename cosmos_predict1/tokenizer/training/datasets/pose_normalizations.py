# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Alternative pose normalization strategies for hyperparameter optimization.
Each strategy normalizes pose data differently to avoid mean pose collapse.
"""

import numpy as np
from pose_format import Pose
from pose_anonymization.data.normalization import normalize_mean_std


def normalize_mean_std_strategy(pose: Pose) -> Pose:
    """
    Strategy 1: Global mean-std normalization (BASELINE - current approach).
    Normalizes using dataset-wide mean and std statistics.

    Pros: Consistent normalization across all videos
    Cons: May lead to mean pose collapse, washes out individual characteristics
    """
    return normalize_mean_std(pose)


def normalize_per_video_strategy(pose: Pose) -> Pose:
    """
    Strategy 2: Per-video mean-std normalization.
    Normalizes each video by its own mean and std.

    Pros: Preserves relative movements within each video better
    Cons: Different scale across videos
    """
    pose_copy = pose.copy()
    data = pose_copy.body.data  # Shape: (T, num_people, num_keypoints, 3) - numpy array

    # Compute mean and std for this specific video
    # Only use valid (non-zero confidence) keypoints
    valid_mask = data[:, :, :, 2] > 0.1  # confidence > 0.1

    # Get valid x,y coordinates
    valid_x = data[:, :, :, 0][valid_mask]
    valid_y = data[:, :, :, 1][valid_mask]

    if valid_x.size > 0 and valid_y.size > 0:
        # Compute mean and std for x and y separately
        mean_x = np.mean(valid_x)
        mean_y = np.mean(valid_y)
        std_x = np.std(valid_x) + 1e-8
        std_y = np.std(valid_y) + 1e-8

        # Normalize x, y coordinates (keep confidence as is)
        pose_copy.body.data[:, :, :, 0] = (data[:, :, :, 0] - mean_x) / std_x
        pose_copy.body.data[:, :, :, 1] = (data[:, :, :, 1] - mean_y) / std_y

    return pose_copy


def normalize_minmax_strategy(pose: Pose) -> Pose:
    """
    Strategy 3: Min-max normalization to [-1, 1] range.
    Simple linear scaling of coordinates.

    Pros: Simple, preserves relative positions
    Cons: Sensitive to outliers, different people have different scales
    """
    pose_copy = pose.copy()
    data = pose_copy.body.data  # Shape: (T, num_people, num_keypoints, 3)

    # Get image dimensions for normalization
    width = float(pose_copy.header.dimensions.width)
    height = float(pose_copy.header.dimensions.height)

    # Normalize to [-1, 1]
    pose_copy.body.data[:, :, :, 0] = (data[:, :, :, 0] / width) * 2.0 - 1.0   # x
    pose_copy.body.data[:, :, :, 1] = (data[:, :, :, 1] / height) * 2.0 - 1.0  # y
    # Keep confidence as is

    return pose_copy


def normalize_first_frame_strategy(pose: Pose) -> Pose:
    """
    Strategy 4: Normalize relative to first frame.
    Subtracts first frame mean position, normalizes by first frame std.

    Pros: Good for capturing motion/changes over time
    Cons: First frame must be representative, issues if first frame has missing keypoints
    """
    pose_copy = pose.copy()
    data = pose_copy.body.data  # Shape: (T, num_people, num_keypoints, 3)

    # Get first frame statistics
    first_frame = data[0, :, :, :]  # (num_people, num_keypoints, 3)
    valid_mask = first_frame[:, :, 2] > 0.1

    # Get valid x,y coordinates from first frame
    valid_x = first_frame[:, :, 0][valid_mask]
    valid_y = first_frame[:, :, 1][valid_mask]

    if valid_x.size > 0 and valid_y.size > 0:
        mean_x = np.mean(valid_x)
        mean_y = np.mean(valid_y)
        std_x = np.std(valid_x) + 1e-8
        std_y = np.std(valid_y) + 1e-8

        # Normalize all frames relative to first frame
        pose_copy.body.data[:, :, :, 0] = (data[:, :, :, 0] - mean_x) / std_x
        pose_copy.body.data[:, :, :, 1] = (data[:, :, :, 1] - mean_y) / std_y
    else:
        # Fallback to minmax if first frame is invalid
        return normalize_minmax_strategy(pose)

    return pose_copy


def normalize_scale_only_strategy(pose: Pose) -> Pose:
    """
    Strategy 5: Scale-only normalization (no centering).
    Scales by person's bounding box size but preserves absolute positions.

    Pros: Preserves spatial relationships better, good for position-aware tasks
    Cons: Not translation invariant
    """
    pose_copy = pose.copy()
    data = pose_copy.body.data  # Shape: (T, num_people, num_keypoints, 3)

    # Compute bounding box across all frames
    valid_mask = data[:, :, :, 2] > 0.1
    if np.any(valid_mask):
        # Get valid coordinates
        valid_x = data[:, :, :, 0][valid_mask]
        valid_y = data[:, :, :, 1][valid_mask]

        if valid_x.size > 0 and valid_y.size > 0:
            # Get bounding box
            bbox_width = np.max(valid_x) - np.min(valid_x) + 1e-8
            bbox_height = np.max(valid_y) - np.min(valid_y) + 1e-8
            bbox_size = max(bbox_width, bbox_height)

            # Scale by bounding box size (no centering)
            pose_copy.body.data[:, :, :, 0] = data[:, :, :, 0] / bbox_size
            pose_copy.body.data[:, :, :, 1] = data[:, :, :, 1] / bbox_size
            # Keep confidence as is
        else:
            # Fallback to minmax if no valid keypoints
            return normalize_minmax_strategy(pose)
    else:
        # Fallback to minmax if no valid keypoints
        return normalize_minmax_strategy(pose)

    return pose_copy


# Dictionary mapping strategy names to functions
NORMALIZATION_STRATEGIES = {
    'mean_std': normalize_mean_std_strategy,
    'per_video': normalize_per_video_strategy,
    'minmax': normalize_minmax_strategy,
    'first_frame': normalize_first_frame_strategy,
    'scale_only': normalize_scale_only_strategy,
}


def normalize_pose(pose: Pose, strategy: str = 'mean_std') -> Pose:
    """
    Normalize pose using the specified strategy.

    Args:
        pose: Pose object to normalize
        strategy: One of: 'mean_std', 'per_video', 'minmax', 'first_frame', 'scale_only'

    Returns:
        Normalized Pose object
    """
    if strategy not in NORMALIZATION_STRATEGIES:
        raise ValueError(f"Unknown normalization strategy: {strategy}. "
                        f"Available: {list(NORMALIZATION_STRATEGIES.keys())}")

    return NORMALIZATION_STRATEGIES[strategy](pose)
