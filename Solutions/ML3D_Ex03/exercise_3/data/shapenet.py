from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)
        print(input_sdf.shape)
        print(target_df.shape)

        # limit input_sdf values between -3 and 3
        input_sdf = np.clip(input_sdf, -self.truncation_distance, self.truncation_distance)
        target_df = np.clip(target_df, 0, self.truncation_distance)

        # add a new first axis to the arrays
        input_sdf = np.expand_dims(input_sdf, axis=0)

        # get the sign of arrays
        input_sdf_sign = np.sign(input_sdf)

        # get the absolute value of arrays
        input_sdf = np.abs(input_sdf)

        # concatenate the sign and absolute value arrays
        input_sdf = np.concatenate((input_sdf, input_sdf_sign), axis=0)
    
        target_df = np.log(target_df + 1)

    
        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch["input_sdf"] = batch["input_sdf"].to(device)
        batch["target_df"] = batch["target_df"].to(device)
        return batch

    @staticmethod
    def get_shape_sdf(shapenet_id):
        path = ShapeNet.dataset_sdf_path / f"{shapenet_id}.sdf"
        # 3 uint64 from file
        dimX, dimY, dimZ = np.fromfile(path, dtype=np.uint64, count=3)
        # read rest of the file as float32
        sdf = np.fromfile(path, dtype=np.float32)[6:]

        sdf = sdf.reshape(dimX, dimY, dimZ)
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        path = ShapeNet.dataset_df_path / f"{shapenet_id}.df"
        # 3 uint64 from file
        dimX, dimY, dimZ = np.fromfile(path, dtype=np.uint64, count=3)
        # read rest of the file as float32
        df = np.fromfile(path, dtype=np.float32)[6:]
        df = df.reshape(dimX, dimY, dimZ)
        return df
