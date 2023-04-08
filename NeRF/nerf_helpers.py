import torch
import torch.nn as nn
import numpy as np

def get_rays(H,W,K,c2w):
    """
    :param K: camera para
    :param c2w: camera matrix to real world matrix
    :return: rays_o, rays_d
    """

    i,j = torch.meshgrid(torch.linspace(0,W-1,W),torch.linspace(0,H-1,H),indexing='ij')
    i = i.t()
    #  dirs : [400,400,3]
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
