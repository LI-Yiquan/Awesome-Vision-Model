from torch.nn import Module

from Model.PVCNN2Base import *
import torch.nn.functional as F


class ShapeEncoder(Module):
    #  (out_channels, num_blocks, voxel_resolution),(256, 0.2, 32, (32, 64))
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 32))),
        ((32, 1, 16), (256, 0.2, 32, (32, 64))),
    ]

    fp_blocks = []
    def __init__(self, embed_dim=0, use_att=False, dropout=0.1,
                 extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        assert embed_dim == 0
        self.in_channels = 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128],
                                          # was 0.5
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        # TODO: complete the following layers

        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(256)
        self.fc1_m = nn.Linear(128*256, 256)
        self.fc2_m = nn.Linear(256, 256)
        self.fc3_m = nn.Linear(256, 128)

        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(256)
        self.fc1_v = nn.Linear(128*256, 256)
        self.fc2_v = nn.Linear(256, 256)
        self.fc3_v = nn.Linear(256, 128)


    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        coords, features = inputs[:, :3, :].contiguous(), inputs
        for i, sa_blocks in enumerate(self.sa_layers):
            features, coords, _ = sa_blocks((features, coords, None))

        features = self.classifier(features)

        features = torch.flatten(features, start_dim=1, end_dim=-1)

        m = F.relu(self.fc_bn1_m(self.fc1_m(features)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(features)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        return m, v
