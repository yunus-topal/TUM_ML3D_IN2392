import numpy as np
import torch
from skimage.measure import marching_cubes

from exercise_3.model.threedepn import ThreeDEPN


class InferenceHandler3DEPN:
    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = ThreeDEPN()
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        self.truncation_distance = 3

    def infer_single(self, input_sdf, target_df):
        """
        Reconstruct a full shape given a partial observation
        :param input_sdf: Input grid with partial SDF of shape 32x32x32
        :param target_df: Target grid with complete DF of shape 32x32x32
        :return: Tuple with mesh representations of input, reconstruction, and target
        """
        # TODO Apply truncation distance: SDF values should lie within -3 and 3, DF values between 0 and 3
        
        # clip input_sdf and target_df to the truncation distance considering input_sdf and target_df has np.inf values
        input_sdf_clip = np.clip(input_sdf, -self.truncation_distance, self.truncation_distance)
        target_df = np.clip(target_df, 0, self.truncation_distance)

        with torch.no_grad():
            reconstructed_df = None
            # TODO: Pass input in the right format though the network and revert the log scaling by applying exp and subtracting 1

            input_sdf = np.expand_dims(input_sdf_clip, axis=0) # 1 x 32 x 32 x 32
            input_sdf_sign = np.sign(input_sdf) # 1 x 32 x 32 x 32
            input_sdf = np.abs(input_sdf) # 1 x 32 x 32 x 32
            input_sdf = np.concatenate((input_sdf, input_sdf_sign), axis=0) # 2 x 32 x 32 x 32
            input_sdf = np.expand_dims(input_sdf, axis=0) # 1 x 2 x 32 x 32 x 32

            reconstructed_df = self.model(torch.from_numpy(input_sdf).to(torch.float32)) # 1 x 32 x 32 x 32

        input_sdf = np.abs(input_sdf_clip)
        input_mesh = marching_cubes(input_sdf, level=1)
        reconstructed_mesh = marching_cubes(reconstructed_df.squeeze(0).numpy(), level=1)
        target_mesh = marching_cubes(target_df, level=1)
        return input_mesh, reconstructed_mesh, target_mesh
