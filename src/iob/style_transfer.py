import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from .utils import image_loader, image_unloader, preserve_color_lab, preserve_color_ycbcr
from .layers import ContentLoss, StyleLoss, Normalization
from .config import *
import copy

class StyleTransfer:
    def __init__(self, content_layers = DEFAULT_CONTENT_LAYERS, style_layers = DEFAULT_STYLE_LAYERS, content_weights = DEFAULT_CONTENT_WEIGHTS, style_weights = DEFAULT_STYLE_WEIGHTS, optimizer = DEFAULT_OPTIMIZER):
        self.device = device
        # Currently only support VGG19
        self.cnn = copy.deepcopy(models.vgg19(pretrained=True).features.to(self.device).eval())
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        # Define content and style layers here
        self.content_layers = content_layers
        self.style_layers = style_layers        
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.optimizer = optimizer
        
    def get_style_model_and_losses(self, style_img, content_img):
        # Normalization module
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(device)

        # Lists to hold content and style loss modules
        content_losses = []
        style_losses = []

        # Sequential model
        model = nn.Sequential(normalization)

        i = 0  # Increment every time a convolutional layer is added
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # In-place ReLU can cause problems, so use out-of-place version
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # Add style loss
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature, weight=self.style_weights[self.style_layers.index(name)])
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

            # Add content loss
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target, weight=self.content_weights[self.content_layers.index(name)])
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

        # Trim the model after the last content and style losses
        for j in range(len(model) - 1, -1, -1):
            if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
                break

        model = model[:j+1]

        return model, style_losses, content_losses

    def get_optimizer(self, input_img, learning_rate=None):
        if learning_rate is None:
            if self.optimizer == 'lbfgs':
                learning_rate = 0.1
            elif self.optimizer == 'adam':
                learning_rate = 0.01
            elif self.optimizer == 'sgd':
                learning_rate = 0.01

        if self.optimizer == 'lbfgs':
            optimizer = optim.LBFGS([input_img], lr=learning_rate)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam([input_img], lr=learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD([input_img], lr=learning_rate)

        return optimizer

    def run_style_transfer(self, content_img_path, style_img_path, imsize = DEFAULT_IMSIZE, initialization=DEFAULT_INITIALIZATION, num_steps=DEFAULT_NUM_STEPS, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, preserve_color = DEFAULT_PRESERVE_COLOR, return_tensor = DEFAULT_RETURN_TENSOR):
        '''
        content_img_path: Path to the content image
        style_img_path: Path to the style image
        initialization (str): 'content' or 'random'
        num_steps (int): Number of optimization steps
        alpha (float): Content weight
        beta (float): Style weight
        preserve_color (bool): Preserve color of content image
        return_tensor (bool): Return tensor instead of PIL image

        return: PIL image or tensor
        '''

        content_img = image_loader(content_img_path, imsize)
        style_img = image_loader(style_img_path, imsize)
        print(style_img.size(), content_img.size())
        assert style_img.size() == content_img.size(), \
            "Style and content images must be the same size"
        if initialization == 'random':
            input_img = torch.randn(content_img.data.size(), device=device)
        else:
            input_img = content_img.clone()
        input_img.requires_grad_(True)
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)
        
        optimizer = self.get_optimizer(input_img)
        
        print('Optimizing...')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # Correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)    
                style_score = 0
                content_score = 0

                # Accumulate style and content losses
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # Total loss
                loss = beta * style_score + alpha * content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("Step {}: Style Loss: {:4f} Content Loss: {:4f}".format(
                        run[0], style_score.item(), content_score.item()))
                return loss

            optimizer.step(closure)

        # Clamp the final output image
        input_img.data.clamp_(0, 1)
        if preserve_color:
            input_img = preserve_color_lab(content_img, input_img)

        if not return_tensor:
            input_img = image_unloader(input_img)

        return input_img