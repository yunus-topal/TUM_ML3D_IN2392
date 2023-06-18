import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.conv1 = nn.Conv3d(2, self.num_features, 4, padding=1, stride=2)
        self.rel1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv3d(self.num_features, self.num_features * 2, 4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm3d(self.num_features * 2)
        self.rel2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv3d(self.num_features * 2, self.num_features * 4, 4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm3d(self.num_features * 4)
        self.rel3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv3d(self.num_features * 4, self.num_features * 8, 4, stride=2)
        self.bn4 = nn.BatchNorm3d(self.num_features * 8)
        self.rel4 = nn.LeakyReLU(0.2)

        # TODO: 2 Bottleneck layers

        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.Linear(self.num_features * 8, self.num_features * 8),
        )

        # TODO: 4 Decoder layers

        self.tran1 = nn.ConvTranspose3d(self.num_features * 8 * 2, self.num_features * 4, 4, stride=2)
        self.bn5 = nn.BatchNorm3d(self.num_features * 4)
        self.rel5 = nn.ReLU()

        self.tran2 = nn.ConvTranspose3d(self.num_features * 8, self.num_features * 2, 4, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(self.num_features * 2)
        self.rel6 = nn.ReLU()

        self.tran3 = nn.ConvTranspose3d(self.num_features * 4, self.num_features, 4, padding=1, stride=2)
        self.bn7 = nn.BatchNorm3d(self.num_features)
        self.rel7 = nn.ReLU()

        self.tran4 = nn.ConvTranspose3d(self.num_features * 2, 1, 4, padding=1, stride=2)

        

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # Reshape and apply bottleneck layers
        x1 = self.conv1(x)
        x1 = self.rel1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.rel2(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.rel3(x3)
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.rel4(x4) # b x 640 x 1 x 1 x 1

        x4_flat = x4.view(b, -1)
        
        x4_flat = self.bottleneck(x4_flat)

        # concatanate x4_flat and x4
        x4_flat = x4_flat.view(b, self.num_features * 8, 1, 1, 1)
        x4 = torch.cat((x4, x4_flat), dim=1)

        x5 = self.tran1(x4)
        x5 = self.bn5(x5)
        x5 = self.rel5(x5)

        # concatanate x5 and x3
        x5 = torch.cat((x5, x3), dim=1)

        x6 = self.tran2(x5)
        x6 = self.bn6(x6)
        x6 = self.rel6(x6)

        # concatanate x6 and x2
        x6 = torch.cat((x6, x2), dim=1)

        x7 = self.tran3(x6)
        x7 = self.bn7(x7)
        x7 = self.rel7(x7)

        # concatanate x7 and x1
        x7 = torch.cat((x7, x1), dim=1)

        x8 = self.tran4(x7)

        x8 = torch.log(torch.abs(x8) + 1)
        # remove the channel dimension
        x8 = x8.squeeze(1)
        return x8
