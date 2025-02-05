import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from lidarnerf.dataset.base_dataset import get_lidar_rays, BaseDataset


@dataclass
class ETHDataset(BaseDataset):
    device: str = "cpu"
    split: str = "train"  # train, val, test
    root_path: str = "data/eth"
    sequence_id: str = "seq10"
    preload: bool = True  # preload data into GPU
    scale: float = (
        1  # camera radius scale to make sure camera are inside the bounding box.
    )
    offset: list = field(default_factory=list)  # offset
    # bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
    fp16: bool = True  # if preload, load into fp16.
    patch_size: int = 1  # size of the image to extract from the scene.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    enable_lidar: bool = True
    num_rays: int = 4096
    num_rays_lidar: int = 4096

    def __post_init__(self):
        print(f"Using sequence {self.sequence_id}")

        self.training = self.split in ["train", "all", "trainval"]
        self.num_rays = self.num_rays if self.training else -1
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        # load nerf-compatible format data.
        with open(
            os.path.join(
                self.root_path, f"transforms_{self.sequence_id}_{self.split}.json"
            ),
            "r",
        ) as f:
            transform = json.load(f)

        # load image size
        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])
        else:
            raise RuntimeError("h_lidar and/or w_lidar not found in JSON")

        # read images
        frames = transform["frames"]

        self.poses_lidar = []
        self.images_lidar = []
        for f in tqdm.tqdm(frames, desc=f"Loading {self.split} data"):
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]

            f_lidar_path = os.path.join(self.root_path, f["lidar_file_path"])

            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path)
            ray_drop = np.where(pc.reshape(-1, 3)[:, 2] == 0.0, 0.0, 1.0).reshape(
                self.H_lidar, self.W_lidar, 1
            )

            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale],
                axis=-1,
            )

            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (
            self.poses_lidar[:, :3, -1] - self.offset
        ) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        if self.images_lidar is not None:
            self.images_lidar = torch.from_numpy(
                np.stack(self.images_lidar, axis=0)
            ).float()  # [N, H, W, C]

        if self.preload:
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.images_lidar is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images_lidar = self.images_lidar.to(dtype).to(self.device)

        self.intrinsics_lidar = (12.5, 120.0)  # fov_up, fov
        self.intrinsics_lidar_hor = (-60, 60)

    def collate(self, index):
        B = len(index)  # a list of length 1

        results = {}

        if self.enable_lidar:
            poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]
            rays_lidar = get_lidar_rays(
                poses_lidar,
                self.intrinsics_lidar,
                self.H_lidar,
                self.W_lidar,
                self.num_rays_lidar,
                self.patch_size_lidar,
                intrinsics_hor=self.intrinsics_lidar_hor
            )

            results.update(
                {
                    "H_lidar": self.H_lidar,
                    "W_lidar": self.W_lidar,
                    "rays_o_lidar": rays_lidar["rays_o"],
                    "rays_d_lidar": rays_lidar["rays_d"],
                }
            )

        if self.images_lidar is not None and self.enable_lidar:
            images_lidar = self.images_lidar[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images_lidar.shape[-1]
                images_lidar = torch.gather(
                    images_lidar.view(B, -1, C),
                    1,
                    torch.stack(C * [rays_lidar["inds"]], -1),
                )  # [B, N, 3/4]
            results["images_lidar"] = images_lidar

        return results

    def dataloader(self):
        size = len(self.poses_lidar)
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)
        return num_frames
