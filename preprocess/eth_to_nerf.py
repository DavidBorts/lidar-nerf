import os
from pathlib import Path
import argparse

from eth_loader import ETHLoader
import camtools as ct
import numpy as np
import json

def main():
    project_root = Path(__file__).parent.parent
    eth_root = project_root / "data" / "eth"

    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_id",
        type=str,
        default="10",
        help="sequence name to use",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        help="(inclusive) start frame idx"
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        help="(inclusive) end frame idx"
    )
    parser.add_argument(
        "--val_frames",
        type=list,
        help="list of val frame idxs"
    )
    args = parser.parse_args()

    # Specify frames and splits.
    sequence_name = args.seq_id
    s_frame_id = args.start_frame
    e_frame_id = args.end_frame  # Inclusive

    eth_sequence_ids = [
        "seq10"
    ]

    if sequence_name not in eth_sequence_ids:
        raise ValueError(
            f"Unknown sequence id {sequence_name}"
        )

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = args.val_frames
    test_frame_ids = sorted([int(test_id) for test_id in test_frame_ids])

    # load in radar, lidar, and gnss frames, filter, and sort
    data_path = eth_root / sequence_name
    lidar_dir = data_path / "lidar"
    lidar_frames = os.listdir(lidar_dir)
    lidar_frames = sorted([f for f in lidar_frames if f.endswith('.bin')])
    gnss_dir = data_path / "gnss"

    # debug prints
    print(f"seq id: {sequence_name}")
    print(f"start frame #, end frame #: {s_frame_id}, {e_frame_id}")
    print(f"selected frame #s {frame_ids}")
    print(f"selected test frame #s: {test_frame_ids}")
    print(f"data path: {data_path}")
    print(f"lidar dir: {lidar_dir}")
    print(f"all lidar frames: {lidar_frames}")
    print(f"gnss dir: {gnss_dir}")
    quit()

    # convert ids to lidar timestamps
    # NOTE: frame idx are 1-indexed
    frame_ids = [lidar_frames[frame_id-1] for frame_id in frame_ids]
    test_frame_ids = [lidar_frames[frame_id-1] for frame_id in test_frame_ids]
    train_frame_ids = [x for x in frame_ids if x not in test_frame_ids]

    # Load KITTI-360 dataset.
    eth = ETHLoader(eth_root, data_path, lidar_dir, gnss_dir)

    # Get lidar paths (range view not raw data).
    range_view_dir = data_path / "train"

    # Get lidar2worlds
    lidar2worlds = eth.load_lidars(sequence_name, train_frame_ids, test_frame_ids)

    # Get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_dir / "{:010d}.npy".format(int(frame_ids[0].split('.')[0])))
    lidar_h, lidar_w, _ = lidar_range_image.shape

    split_to_all_indices = {
        "train": train_frame_ids,
        "val": test_frame_ids,
        "test": test_frame_ids,
    }

    for split, lidar_paths in split_to_all_indices.items():
        print(f"Split {split} has {len(lidar_paths)} frames.")

        range_view_paths = [
            range_view_dir / "{:010d}.npy".format(int(frame_id.split('.')[0])) for frame_id in lidar_paths
        ]

        json_dict = {
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "aabb_scale": 2,
            "frames": [
                {
                    "lidar_file_path": str(
                        range_view_path.relative_to(eth_root)
                    ),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    range_view_path,
                    lidar2world,
                ) in zip(
                    range_view_paths,
                    lidar2worlds[split],
                )
            ],
        }
        json_path = eth_root / f"transforms_{sequence_name}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")

if __name__ == "__main__":
    main()
