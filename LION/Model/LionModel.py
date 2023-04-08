import numpy as np
import torch
from torch import nn
from Model.Decoder import Decoder
from Model.PointDDM import PointDDM
from Model.PointEncoder import PointEncoder
from Model.ShapeDDM import ShapeDDM
from Model.ShapeEncoder import ShapeEncoder
from Model.PVCNN2Base import *
from Model.GaussianDiffusion import *

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


class LionModel(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str):
        super(LionModel, self).__init__()

        # TODO: use different attention method in different layer
        self.shape_encoder = ShapeEncoder(embed_dim=0, use_att=False, dropout=0.1,
                                          extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1)
        self.point_encoder = PointEncoder(embed_dim=0, use_att=False, dropout=0.1,
                                          extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1)
        self.decoder = Decoder(embed_dim=0, use_att=True, dropout=0.1,
                               extra_feature_channels=0, width_multiplier=1, voxel_resolution_multiplier=1)

        self.point_ddm = PointDDM(args, betas, loss_type, model_mean_type, model_var_type)
        self.shape_ddm = ShapeDDM(args, betas, loss_type, model_mean_type, model_var_type)

        self.AdaMLP1 = nn.Linear(128, 2048)
        self.AdaMLP2 = nn.Linear(128, 2048)
        # self.AdaNorm = SharedMLP(3, 3)

    def get_loss_vae(self, data):
        shape_latent, point_latent, shape_latent_v, point_latent_v = self.get_latent(data)

        output = self.decoder(point_latent, shape_latent)

        shape_entropy = gaussian_entropy(shape_latent_v)

        point_entropy = gaussian_entropy(point_latent_v)

        recon_loss = data - output
        # TODO: complete reconstruction loss, try to use laplace loss for point latent

        loss = recon_loss.mean() + shape_entropy.mean() + point_entropy.mean()

        return loss

    def get_latent(self, data):
        shape_latent_m, shape_latent_v = self.shape_encoder(data)

        shape_latent = reparameterize_gaussian(shape_latent_m, shape_latent_v)

        point_latent_m, point_latent_v = self.point_encoder(data, shape_latent)

        point_latent = reparameterize_gaussian(point_latent_m, point_latent_v)

        return shape_latent, point_latent, shape_latent_v, point_latent_v

    def get_loss_shape_ddm(self, data, noises=None):
        shape_latent, point_latent, _, _ = self.get_latent(data)

        return self.shape_ddm.get_loss_iter(point_latent, noises)

    def get_loss_point_ddm(self, data, noises=None):
        shape_latent, point_latent, _, _ = self.get_latent(data)

        mu = self.AdaMLP1(shape_latent)
        sigma = self.AdaMLP2(shape_latent)
        mu = mu.unsqueeze(1).expand(-1, point_latent.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, point_latent.shape[1], -1)
        inputs = point_latent * sigma + mu

        return self.point_ddm.get_loss_iter(point_latent, noises)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
