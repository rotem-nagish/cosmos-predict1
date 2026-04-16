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

"""
Dataset that takes 128×512 grid videos (4 × 128×128 panels stacked vertically)
as encoder input and the corresponding original dictio*.mp4 videos as the
reconstruction target.

The 4-panel vertical layout is rearranged into a 256×256 square (2×2 grid):
    [panel_0 | panel_1]
    [panel_2 | panel_3]

Run interactively with:
    PYTHONPATH=$(pwd) python cosmos_predict1/tokenizer/training/datasets/grid_video_dataset.py
"""

import os
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs, reduce_holistic
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.append(os.getcwd())
from cosmos_predict1.diffusion.utils.dataset_utils import ToTensorVideo
from cosmos_predict1.tokenizer.training.datasets.pose_normalizations import normalize_pose

BASE_DIR = (
    "/mnt/rylo-tnas/users/rotem/sign/data"
    if "/mnt/rylo-tnas" in os.getcwd()
    else "/workspace/datasets"
)

# ── Hand orientation from pose ──────────────────────────────────────────
# Hand landmark indices within each 21-point MediaPipe hand:
_WRIST, _INDEX_MCP, _PINKY_MCP = 0, 5, 17

# After reduce_holistic + pose_hide_legs (178 keypoints):
#   body: 0-7, face: 8-135, left_hand: 136-156, right_hand: 157-177
_LEFT_HAND_OFFSET = 136
_RIGHT_HAND_OFFSET = 157


def compute_hand_orientation_classes(pose_data: np.ndarray) -> np.ndarray:
    """Compute per-frame hand orientation class (0-5) from raw 3D pose landmarks.

    Uses the palm normal (cross product of wrist→index_mcp and wrist→pinky_mcp)
    to classify into 6 ISWA-style orientation bins:
        0-2: palm facing camera (front), 3 rotation bins
        3-5: palm facing away (back), 3 rotation bins

    Args:
        pose_data: [T, num_keypoints, 3] raw (un-normalised) pose data.
    Returns:
        [T, 2] int array — orientation class per frame for [left, right] hand.
        -1 indicates invalid/missing detection.
    """
    T = pose_data.shape[0]
    result = np.full((T, 2), -1, dtype=np.int64)

    for hand_idx, offset in enumerate([_LEFT_HAND_OFFSET, _RIGHT_HAND_OFFSET]):
        wrist = pose_data[:, offset + _WRIST]          # [T, 3]
        index_mcp = pose_data[:, offset + _INDEX_MCP]   # [T, 3]
        pinky_mcp = pose_data[:, offset + _PINKY_MCP]   # [T, 3]

        v1 = index_mcp - wrist  # [T, 3]
        v2 = pinky_mcp - wrist  # [T, 3]

        # For left hand, flip cross product to get consistent "outward" normal
        normal = np.cross(v1, v2)  # [T, 3]
        if hand_idx == 0:  # left hand
            normal = -normal

        norm_mag = np.linalg.norm(normal, axis=-1, keepdims=True)

        for t in range(T):
            if norm_mag[t, 0] < 1e-6:
                continue  # degenerate (missing or colinear points)

            # Check for NaN
            if np.any(np.isnan(normal[t])):
                continue

            n = normal[t] / norm_mag[t, 0]

            # Front (palm toward camera) vs back: z < 0 means toward camera
            # in MediaPipe coords (z decreases toward camera)
            front = n[2] < 0

            # Rotation bin from xy-plane angle of normal
            angle = np.arctan2(n[1], n[0])
            if -np.pi / 3 <= angle <= np.pi / 3:
                rotation = 0  # straight
            elif angle > np.pi / 3:
                rotation = 1
            else:
                rotation = 2

            result[t, hand_idx] = rotation if front else rotation + 3

    return result


class GridVideoDataset(TorchDataset):
    """Loads 128×512 grid videos (4 stacked 128×128 panels) as encoder input and
    the matching original dictio*.mp4 videos as reconstruction targets.

    Returns a dict with:
        video        : [C, T, 256, 256]  uint8  – 2×2 rearranged grid (encoder input)
        target_video : [C, T, 256, 256]  uint8  – original video resized to 256×256
        loss_mask    : [C, T, 256, 256]  float  – bounding-box mask for the original video
        gt_pose      : Pose object or []
        video_name   : dict with path metadata
        fps, num_frames, padding_mask
    """

    def __init__(
        self,
        grid_video_pattern: str,
        original_video_dir: str,
        sequence_interval: int = 1,
        start_frame_interval: int = 1,
        num_video_frames: int = 49,
        pose_normalization: str = "minmax",
        limit=None,
        target_size: tuple = (256, 256),
        part_fusion: bool = False,
        fliplr_list_path: str = None,
        mask_type: str = 'bb',
    ):
        super().__init__()
        self.grid_video_pattern = grid_video_pattern
        self.original_video_dir = original_video_dir
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_video_frames
        self.pose_normalization = pose_normalization
        self.target_size = target_size  # (H, W) for both output videos
        self.part_fusion = part_fusion  # channel-stack parts instead of 2×2 grid
        self.mask_type = mask_type

        # Set of video basenames that should be horizontally flipped (left-handed signers)
        self.fliplr_set = set()
        if fliplr_list_path is not None:
            with open(fliplr_list_path) as f:
                self.fliplr_set = {line.strip() for line in f if line.strip()}

        self.video_paths = sorted(glob(str(grid_video_pattern)))
        if limit is not None:
            self.video_paths = self.video_paths[:limit]

        print(f"{len(self.video_paths)} grid videos found")
        print(f"Pose normalization: {self.pose_normalization}")

        metadata_path = (
            "/mnt/rylo-tnas/users/rotem/sign/data/video_list.csv"
            if "/mnt/rylo-tnas" in os.getcwd()
            else "/workspace/datasets/video_list.csv"
        )
        metadata = pd.read_csv(metadata_path)
        self.vid2md5 = {row.name: row.md5Hash for row in metadata.itertuples()}

        self.preprocess = T.Compose([ToTensorVideo()])

        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(
            self.samples, key=lambda x: (x["grid_video_path"], x["frame_ids"][0])
        )
        print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0

    # ------------------------------------------------------------------
    # Sample initialisation
    # ------------------------------------------------------------------

    def _init_samples(self, video_paths):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            futures = {
                executor.submit(self._load_and_process_video_path, vp): vp
                for vp in video_paths
            }
            for future in tqdm(as_completed(futures), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    def _load_and_process_video_path(self, grid_video_path):
        try:
            original_path = self._get_original_path(grid_video_path)
            if not os.path.exists(original_path):
                return []

            vr = VideoReader(grid_video_path, ctx=cpu(0), num_threads=0)
            n_frames = len(vr)

            samples = []
            for frame_i in range(0, n_frames, self.start_frame_interval):
                sample = {
                    "grid_video_path": grid_video_path,
                    "original_video_path": original_path,
                    "frame_ids": [],
                }
                curr = frame_i
                while curr <= (n_frames - 1):
                    sample["frame_ids"].append(curr)
                    if len(sample["frame_ids"]) == self.sequence_length:
                        break
                    curr += self.sequence_interval
                if len(sample["frame_ids"]) == self.sequence_length:
                    samples.append(sample)
        except Exception:
            samples = []
        return samples

    def _get_original_path(self, grid_video_path: str) -> str:
        """Map grid video path → corresponding original video path."""
        basename = os.path.basename(grid_video_path).replace("_grid.mp4", ".mp4")
        return os.path.join(self.original_video_dir, basename)

    # ------------------------------------------------------------------
    # Grid-reshape helper
    # ------------------------------------------------------------------

    @staticmethod
    def reshape_grid_frame(frame: np.ndarray) -> np.ndarray:
        """Rearrange a single 128×512 frame (H=512, W=128) into 256×256.

        Vertical layout (4 × 128×128 panels stacked):
            panel_0  (rows   0-127)
            panel_1  (rows 128-255)
            panel_2  (rows 256-383)
            panel_3  (rows 384-511)

        Target 2×2 layout:
            [panel_0 | panel_1]
            [panel_2 | panel_3]
        """
        H, W, C = frame.shape
        ph = H // 4  # 128
        p0 = frame[0 * ph : 1 * ph]
        p1 = frame[1 * ph : 2 * ph]
        p2 = frame[2 * ph : 3 * ph]
        p3 = frame[3 * ph : 4 * ph]
        top = np.concatenate([p0, p1], axis=1)    # [128, 256, C]
        bot = np.concatenate([p2, p3], axis=1)    # [128, 256, C]
        return np.concatenate([top, bot], axis=0)  # [256, 256, C]

    @staticmethod
    def split_panels(frame: np.ndarray) -> list[np.ndarray]:
        """Split a 128×512 frame into 4 separate 128×128 panels.

        Returns [panel_0, panel_1, panel_2, panel_3] where:
            panel_0 = original, panel_1 = face,
            panel_2 = left hand, panel_3 = right hand.
        """
        H, W, C = frame.shape
        ph = H // 4
        return [frame[i * ph : (i + 1) * ph] for i in range(4)]

    # ------------------------------------------------------------------
    # Video loading helpers
    # ------------------------------------------------------------------

    def _load_grid_frames(self, path: str, frame_ids: list) -> np.ndarray:
        """Load grid video and reshape each frame to 256×256. Returns [T,256,256,C].

        When ``self.part_fusion`` is True, each panel is resized to target_size
        independently and concatenated channel-wise, returning [T, H, W, C*4].
        """
        vr = VideoReader(path, ctx=cpu(0), num_threads=0)
        vr.seek(0)
        raw = vr.get_batch(frame_ids).asnumpy()  # [T, 512, 128, C]

        if not self.part_fusion:
            return np.stack([self.reshape_grid_frame(f) for f in raw], axis=0)

        # Part-fusion: resize each panel to target_size and stack channels
        target_h, target_w = self.target_size
        result = []
        for f in raw:
            panels = self.split_panels(f)  # 4 × [128, 128, 3]
            resized = []
            for p in panels:
                r = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                resized.append(r)
            # [H, W, 12]
            result.append(np.concatenate(resized, axis=-1))
        return np.stack(result, axis=0)

    def _load_original_frames(self, path: str, frame_ids: list) -> np.ndarray:
        """Load original video frames, clamping ids if the video is shorter."""
        vr = VideoReader(path, ctx=cpu(0), num_threads=0)
        n = len(vr)
        ids = [min(fid, n - 1) for fid in frame_ids]
        vr.seek(0)
        return vr.get_batch(ids).asnumpy()  # [T, H, W, C]

    def _to_tensor_CTHW(self, frames: np.ndarray) -> torch.Tensor:
        """[T,H,W,C] uint8 → [C,T,H,W] uint8 tensor."""
        t = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2)  # [T,C,H,W]
        t = self.preprocess(t)
        t = torch.clamp(t * 255.0, 0, 255).to(torch.uint8)
        return t.permute(1, 0, 2, 3)  # [C,T,H,W]

    @staticmethod
    def _resize_CTHW(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize [C,T,H,W] uint8 tensor to (target_h, target_w)."""
        C, T, H, W = video.shape
        if H == target_h and W == target_w:
            return video
        v = video.float().permute(1, 0, 2, 3)  # [T,C,H,W]
        v = F.interpolate(v, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return v.permute(1, 0, 2, 3).to(torch.uint8)  # [C,T,H,W]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            grid_path = sample["grid_video_path"]
            orig_path = sample["original_video_path"]
            frame_ids = sample["frame_ids"]
            target_h, target_w = self.target_size

            # ---- Grid video (encoder input) ----
            grid_frames = self._load_grid_frames(grid_path, frame_ids)
            grid_tensor = self._to_tensor_CTHW(grid_frames)          # [C,T,256,256]
            grid_tensor = self._resize_CTHW(grid_tensor, target_h, target_w)

            # ---- Original video (reconstruction target) ----
            orig_frames = self._load_original_frames(orig_path, frame_ids)
            orig_tensor = self._to_tensor_CTHW(orig_frames)          # [C,T,H,W]
            orig_tensor = self._resize_CTHW(orig_tensor, target_h, target_w)

            # Horizontal flip for left-handed signers
            video_basename = os.path.basename(orig_path)
            if video_basename in self.fliplr_set:
                grid_tensor = torch.flip(grid_tensor, dims=[-1])  # flip W axis
                orig_tensor = torch.flip(orig_tensor, dims=[-1])

            data = {
                "video": grid_tensor,
                "target_video": orig_tensor,
                "video_name": {
                    "video_path": grid_path,
                    "original_path": orig_path,
                    "start_frame_id": str(frame_ids[0]),
                },
                "fps": 24,
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, 704, 1280),
            }

            # ---- Pose (from original video) ----
            base_path = f"sign-tube/{os.path.basename(orig_path)}"
            try:
                md5 = self.vid2md5[base_path]
                pose_root = (
                    "/mnt/nas/GCS/sign-mediapipe-holistic-poses"
                    if "/mnt/rylo-tnas" in os.getcwd()
                    else "/workspace/poses"
                )
                pose_path = os.path.join(pose_root, md5 + ".pose")
                with open(pose_path, "rb") as f:
                    gt_pose = Pose.read(f)
                gt_pose = pose_hide_legs(reduce_holistic(gt_pose))

                # Compute hand orientation from raw (un-normalised) pose
                # before normalization distorts the 3D geometry.
                raw_slice = gt_pose.body.data[
                    frame_ids[0] : frame_ids[0] + self.sequence_length, 0
                ]  # [T, num_kp, 3]
                if hasattr(raw_slice, "data"):
                    raw_np = raw_slice.data  # unmask
                else:
                    raw_np = np.array(raw_slice)
                hand_orient = compute_hand_orientation_classes(raw_np)
                data["hand_orientation_gt"] = torch.from_numpy(hand_orient)  # [T, 2] int64

                gt_pose = normalize_pose(gt_pose, strategy=self.pose_normalization)
                gt_pose.body.data = gt_pose.body.data[
                    frame_ids[0] : frame_ids[0] + self.sequence_length
                ]
                gt_pose.body.confidence = gt_pose.body.confidence[
                    frame_ids[0] : frame_ids[0] + self.sequence_length
                ]
                data["gt_pose"] = gt_pose
            except Exception:
                print(f"No pose found for {base_path}")
                data["gt_pose"] = []

            # ---- Crop bounding boxes ----
            # Used by part fusion (positional encoding) and hand_location aux task.
            bb_path = grid_path.replace(".mp4", "_bboxes.npy")
            if os.path.exists(bb_path):
                all_bbs = np.load(bb_path)  # [T_total, 3, 4]
                crop_bbs = all_bbs[frame_ids]  # [T, 3, 4]
                data["crop_bboxes"] = torch.from_numpy(crop_bbs)  # float32
                # Hand location GT: left hand (idx 1) + right hand (idx 2)
                data["hand_location_gt"] = torch.from_numpy(
                    crop_bbs[:, 1:3]  # [T, 2, 4]
                )
            elif self.part_fusion:
                data["crop_bboxes"] = torch.full(
                    (self.sequence_length, 3, 4), -1.0
                )

            # ---- Loss mask (from original video) ----
            data["loss_mask"] = torch.ones_like(orig_tensor, dtype=torch.float32)
            mask_path = (
                orig_path.replace("/processed/", f"/{self.mask_type}_masks/")[: -len(".mp4")] + ".pt"
            )
            if os.path.exists(mask_path):
                loaded = torch.load(mask_path)[
                    frame_ids[0] : frame_ids[0] + self.sequence_length
                ]
                mh, mw = loaded.shape[-2:]
                if mh != target_h or mw != target_w:
                    loaded = F.interpolate(
                        loaded.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                loaded = torch.where(
                    loaded == 0,
                    torch.tensor(0.2, device=loaded.device),
                    loaded / 255,
                )
                if len(loaded) < self.sequence_length:
                    f, h, w = loaded.size()
                    pad = torch.zeros(
                        (self.sequence_length - f, h, w),
                        device=loaded.device,
                        dtype=loaded.dtype,
                    )
                    loaded = torch.cat([loaded, pad], dim=0)
                data["loss_mask"] = loaded.unsqueeze(0).repeat(3, 1, 1, 1)
            else:
                print(f"No bb mask found for {base_path}")

            return data

        except Exception:
            warnings.warn(
                f"Invalid data at index {index} "
                f"({self.samples[index]['grid_video_path']}). Skipping."
            )
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = GridVideoDataset(
        grid_video_pattern=f"{BASE_DIR}/128x512_grid_videos/dictio*.mp4",
        original_video_dir=f"{BASE_DIR}/processed/sign-tube/videos",
        num_video_frames=49,
    )
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Keys:", list(sample.keys()))
        print("video shape:", sample["video"].shape)
        print("target_video shape:", sample["target_video"].shape)
        print("loss_mask shape:", sample["loss_mask"].shape)