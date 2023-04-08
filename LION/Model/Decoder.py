import torch
from torch.nn import Module
from torch import nn
from Model.PVCNN2Base import *


class Decoder(Module):

    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 1, 16), (256, 0.2, 32, (64, 128))),
        ((128, 1, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (128, 128, 128))),
    ]
    fp_blocks = [
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 3, 8)),
        ((128, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = 0
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True,
            embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, 3],
                                          # was 0.5
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.AdaMLP1 = nn.Linear(128, 2048)
        self.AdaMLP2 = nn.Linear(128, 2048)
        # self.AdaNorm = SharedMLP((3 + extra_feature_channels), (3 + extra_feature_channels))



    def forward(self, inputs, shape_latent):

        mu = self.AdaMLP1(shape_latent)
        sigma = self.AdaMLP2(shape_latent)
        mu = mu.unsqueeze(1).expand(-1, inputs.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, inputs.shape[1], -1)
        inputs = inputs * sigma + mu

        # inputs : [B, in_channels + S, N]
        shape_latent_emb = shape_latent.unsqueeze(-1).expand(-1, -1, inputs.shape[-1])

        coords, features = inputs[:, :3, :].contiguous(), inputs

        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, _ = sa_blocks((features, coords, None))

        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):

            features, coords, shape_latent_emb = fp_blocks((coords_list[-1 - fp_idx], coords,features,
                                             in_features_list[-1 - fp_idx],None))

        features = self.classifier(features)

        return features