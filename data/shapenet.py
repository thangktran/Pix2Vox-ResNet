from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # Apply truncation to sdf and df
        input_sdf = np.clip(input_sdf, a_min=-self.truncation_distance, a_max=self.truncation_distance)
        target_df = np.clip(target_df, a_min=0, a_max=self.truncation_distance)
        # Stack (distances, sdf sign) for the input sdf
        input_sdf = np.stack([np.abs(input_sdf), np.sign(input_sdf)])
        # Log-scale target df
        target_df = np.log(np.add(target_df, 1.0))
        
        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # add code to move batch to device
        batch["input_sdf"] = batch["input_sdf"].to(device)
        batch["target_df"] = batch["target_df"].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # implement sdf data loading
        file_path = f'{ShapeNet.dataset_sdf_path}/{shapenet_id}.sdf'
        with open(file_path, "rb") as f:
            dims = np.fromfile(f, dtype=np.uint64, count=3, offset=0)
            num_points = dims[0] * dims[1] * dims[2]
            sdf = np.fromfile(f, dtype=np.float32, count=num_points, offset=0)
            sdf = sdf.reshape((dims[0], dims[1], dims[2]))
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # implement df data loading
        file_path = f'{ShapeNet.dataset_df_path}/{shapenet_id}.df'
        with open(file_path, "rb") as f:
            dims = np.fromfile(f, dtype=np.uint64, count=3, offset=0)
            num_points = dims[0] * dims[1] * dims[2]
            sdf = np.fromfile(f, dtype=np.float32, count=num_points, offset=0)
            sdf = sdf.reshape((dims[0], dims[1], dims[2]))
        return sdf
        return df
