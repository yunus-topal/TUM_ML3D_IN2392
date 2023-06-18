import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.lin1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(latent_size + 3, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 253)),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )
        # concatenate

        self.lin2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(512, 1)
        )
        

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x1 = self.lin1(x_in)

        # concat x1 and x_in
        x2 = torch.cat((x1, x_in), dim=1)

        x3 = self.lin2(x2)
        return x3
