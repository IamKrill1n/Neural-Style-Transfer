from collections import namedtuple

import torch
from torchvision import models
from .config import *

class Vgg16(torch.nn.Module):
    def __init__(self, content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS, requires_grad=False):
        super(Vgg16, self).__init__()
        self.vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval().features
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.selected_layers = sorted(set(self.content_layers + self.style_layers))
        self.layer_mapping = vgg16_layers_mapping
        self.selected_indices = [self.layer_mapping[name] for name in self.selected_layers]
        self.index_mapping = {idx: name for name, idx in self.layer_mapping.items()}
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def get_layer_name(self, idx):
        return self.index_mapping[idx] 

    def forward(self, x):
        features = {}
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)
        x = (x - mean) / std
        i = 0
        for layer in self.vgg_pretrained_features:
            x = layer(x)
            layer_name = self.get_layer_name(i)
            if layer_name in self.selected_layers:
                features[layer_name] = x
            i += 1
            if i > self.selected_indices[-1]:
                break

        outputs = {name: features[name] for name in self.selected_layers}
        return outputs
    
class Vgg19(Vgg16):
    def __init__(self, content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS, requires_grad=False):
        super(Vgg19, self).__init__(content_layers, style_layers, requires_grad)
        self.vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).eval().features
        self.layer_mapping = vgg19_layers_mapping