import torch
import torch.optim as optim
from .utils import image_loader, image_unloader, preserve_color_lab, preserve_color_ycbcr
from .config import *
from .perceptual_loss_net import Vgg16, Vgg19
from .loss import content_loss, style_loss, total_variation_loss

class StyleTransfer:
    def __init__(self, cnn = DEFAULT_CNN, content_layers = DEFAULT_CONTENT_LAYERS, style_layers = DEFAULT_STYLE_LAYERS, content_weights = DEFAULT_CONTENT_WEIGHTS, style_weights = DEFAULT_STYLE_WEIGHTS, optimizer = DEFAULT_OPTIMIZER):
        self.device = device
        # Define content and style layers here
        self.content_layers = content_layers
        self.style_layers = style_layers      
        # Currently only support VGG19
        if cnn == 'vgg19':
            self.cnn = Vgg19(content_layers=self.content_layers, style_layers=self.style_layers).to(self.device)
        elif cnn == 'vgg16':
            self.cnn = Vgg16(content_layers=self.content_layers, style_layers=self.style_layers).to(self.device)
        else:
            raise ValueError(str(cnn) + " is not supported. Only 'vgg19' and 'vgg16' are supported")  
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.optimizer = optimizer

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
        else:
            raise ValueError("Only 'lbfgs', 'adam', and 'sgd' are supported")
        
        return optimizer

    def stylize(self, content_img_path, style_img_path, imsize = DEFAULT_IMSIZE, initialization=DEFAULT_INITIALIZATION, num_steps=DEFAULT_NUM_STEPS, learning_rate = None, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, tv_weight = DEFAULT_TV_WEIGHT, preserve_color = DEFAULT_PRESERVE_COLOR, return_tensor = DEFAULT_RETURN_TENSOR):
        '''
        content_img_path: Path to the content image
        style_img_path: Path to the style image
        imsize (int or tuple): Size of the output image
        initialization (str): 'content' or 'random'
        num_steps (int): Number of optimization steps
        lr (float): Learning rate of the optimizer
        alpha (float): Content weight
        beta (float): Style weight
        tv_weight (float): Total variation weight
        preserve_color (bool): Preserve color of content image
        return_tensor (bool): Return tensor instead of PIL image

        return: PIL image or tensor
        '''

        content_img = image_loader(content_img_path, imsize)
        style_img = image_loader(style_img_path, content_img.shape[2:])
        print(style_img.size(), content_img.size())
        assert style_img.size() == content_img.size(), \
            "Style and content images must be the same size"
        if initialization == 'random':
            input_img = torch.randn(content_img.data.size(), device=device)
        else:
            input_img = content_img.clone()
        input_img.requires_grad_(True)
        
        
        optimizer = self.get_optimizer(input_img, learning_rate=learning_rate)
        content_target = self.cnn(content_img)
        style_target = self.cnn(style_img)

        print('Optimizing...')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                outputs = self.cnn(input_img)   
                content_score = 0
                style_score = 0

                for layer, weight in zip(self.content_layers, self.content_weights):
                    content_score += weight * content_loss(outputs[layer], content_target[layer].detach())

                for layer, weight in zip(self.style_layers, self.style_weights):
                    style_score += weight * style_loss(outputs[layer], style_target[layer].detach())

                loss = alpha * content_score + beta * style_score + tv_weight * total_variation_loss(input_img)
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("Step {}: Style Loss: {:4f} Content Loss: {:4f}".format(
                        run[0], style_score.item(), content_score.item()))
                return loss

            optimizer.step(closure)

        # Clamp the final output image
        with torch.no_grad():
            input_img.data.clamp_(0, 1)
            
        if preserve_color:
            input_img = preserve_color_lab(content_img, input_img)

        if not return_tensor:
            input_img = image_unloader(input_img)

        return input_img