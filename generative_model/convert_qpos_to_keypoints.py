import argparse
import h5py
import torch

from src.utils.keypoints_utils import HandModelKeypoints
from src.utils.hdf5_utils import append_to_hdf5


def main():
    parser = argparse.ArgumentParser(description="Convert qpos in an HDF5 file to FK keypoints.")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path} ...")
    with h5py.File(args.data_path, 'r') as f:
        if 'qpos' not in f:
            raise KeyError("Dataset 'qpos' not found.")
        if 'keypoints' in f:
            raise KeyError("Dataset 'keypoints' already exists. Remove it first.")
        num_samples = f['qpos'].shape[0]

    print("Converting qpos to keypoints ...")
    hand_model = HandModelKeypoints()

    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)
        with h5py.File(args.data_path, 'r') as f:
            qpos_batch = f['qpos'][start:end]
        qpos_tensor = torch.tensor(qpos_batch, dtype=torch.float32, device='cuda')
        keypoints_batch = hand_model.compute_keypoints(qpos_tensor).cpu().numpy()
        append_to_hdf5(args.data_path, {"keypoints": keypoints_batch})
        print(f"Processed {end}/{num_samples} samples.")

    with h5py.File(args.data_path, 'r') as f:
        if f['keypoints'].shape[0] != num_samples:
            raise RuntimeError("Mismatch: keypoints count != qpos count.")

    print("Done.")


if __name__ == "__main__":
    main()
