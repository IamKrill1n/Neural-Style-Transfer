import torch
import torch.nn as nn
from .config import device

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # Detach target content from the graph
        self.target = target.detach() * weight
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input * self.weight, self.target)
        return input
    
def gram_matrix(input):
    batch_size, feature_maps, h, w = input.size()
    features = input.view(batch_size * feature_maps, h * w)
    G = torch.mm(features, features.t())
    # Normalize the Gram Matrix
    return G.div(batch_size * feature_maps * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature, weight):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach() * weight
        self.weight = weight
        self.loss = 0
    
    def forward(self, input):
        G = gram_matrix(input) * self.weight
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape mean and std to [C x 1 x 1] for broadcasting
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        # Normalize the image
        return (img - self.mean) / self.std