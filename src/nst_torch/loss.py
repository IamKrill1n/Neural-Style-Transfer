import torch
import torch.nn as nn
from .config import device
import torch.nn.functional as F

def content_loss(input, target):
    return F.mse_loss(input, target)
    
def gram_matrix(input):
    batch_size, feature_maps, h, w = input.size()
    features = input.view(batch_size * feature_maps, h * w)
    G = torch.mm(features, features.t())
    # Normalize the Gram Matrix
    return G.div(batch_size * feature_maps * h * w)

def style_loss(input, target):
    G_input = gram_matrix(input)
    G_target = gram_matrix(target)
    return F.mse_loss(G_input, G_target)

def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w