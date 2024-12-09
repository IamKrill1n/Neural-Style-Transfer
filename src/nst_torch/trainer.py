import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from .config import *
from .loss import content_loss, style_loss, total_variation_loss, gram_matrix
from .utils import image_loader, image_unloader, get_content_loader, imshow, preserve_color_lab
from .transformer_net import TransformerNet
from .perceptual_loss_net import Vgg16, Vgg19
from .style_transfer import StyleTransfer

class FastStyleTransferTrainer(StyleTransfer):
    def __init__(self, cnn = DEFAULT_CNN, content_layers = DEFAULT_CONTENT_LAYERS, style_layers = DEFAULT_STYLE_LAYERS, content_weights = DEFAULT_CONTENT_WEIGHTS, style_weights = DEFAULT_STYLE_WEIGHTS, optimizer = DEFAULT_FAST_OPTIMIZER):
        super().__init__(cnn, content_layers, style_layers, content_weights, style_weights, optimizer)
        self.transformer = TransformerNet().to(self.device)

    def get_optimizer(self, model, learning_rate):
        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE
        if self.optimizer == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif self.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

    def train(self, style_image_path, content_image_path, imsize = DEFAULT_TRAIN_IMSIZE, alpha = DEFAULT_ALPHA, beta = DEFAULT_BETA, tv_weight = DEFAULT_TV_WEIGHT, epochs=2, batch_size=4, learning_rate=None, checkpoint_path=None):
        self.transformer.train()
        # Load images
        train_loader = get_content_loader(content_image_path, batch_size=batch_size, imsize=imsize)
        style_image = image_loader(style_image_path, imsize=imsize).to(self.device)
        # Compute style features without repeating
        with torch.no_grad():
            style_features = self.cnn(style_image)
            # Compute Gram matrices for style features
            style_grams = {layer: gram_matrix(style_features[layer]) for layer in self.style_layers}
        # Define optimizer
        optimizer = self.get_optimizer(self.transformer, learning_rate)
        
        for e in range(epochs):
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            for batch_id, content_batch in enumerate(train_loader):
                count += len(content_batch)
                optimizer.zero_grad()
                content_score = 0.
                style_score = 0.
                
                content_batch = content_batch.to(self.device)
                # Stylized image using transformer network
                stylized_batch = self.transformer(content_batch)
                # Extract features from perceptual loss network
                content_batch_features = self.cnn(content_batch)
                stylized_batch_features = self.cnn(stylized_batch)

                # Compute content loss
                for layer, weight in zip(self.content_layers, self.content_weights):
                    content_score += weight * content_loss(stylized_batch_features[layer], content_batch_features[layer].detach())
                
                # Compute style loss
                for layer, weight in zip(self.style_layers, self.style_weights):
                    # Compute Gram matrix of the stylized features
                    stylized_gram = gram_matrix(stylized_batch_features[layer])

                    # Expand the style Gram matrix to match batch size
                    style_gram = style_grams[layer]
                    style_gram_batch = style_gram.unsqueeze(0).expand(stylized_gram.size(0), -1, -1)

                    # Compute style loss between the Gram matrices
                    style_score += weight * style_loss(
                        stylized_gram, style_gram_batch.detach()
                    )

                loss = alpha * content_score + beta * style_score + tv_weight * total_variation_loss(stylized_batch)
                loss.backward()
                optimizer.step()

                agg_content_loss += content_score.item()
                agg_style_loss += style_score.item()
                
            
            print(f"Epoch {e+1}/{epochs}, Content Loss: {agg_content_loss/count}, Style Loss: {agg_style_loss/count}")

            if checkpoint_path is not None:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                transformer_cpu = self.transformer.to('cpu').eval()
                ckpt_model_filename = f"ckpt_epoch_{e}.pth"
                ckpt_model_path = os.path.join(checkpoint_path, ckpt_model_filename)
                torch.save(transformer_cpu.state_dict(), ckpt_model_path)
                self.transformer.to(self.device).train()
        

    def stylize(self, content_image, preserve_color = False, return_tensor=True):
        self.transformer.eval()
        with torch.no_grad():
            content_image = content_image.to(self.device)
            stylized_image = self.transformer(content_image)
            stylized_image.data.clamp_(0, 1)

        if preserve_color:
            stylized_image  = preserve_color_lab(content_image, stylized_image)

        if not return_tensor:
            stylized_image = image_unloader(stylized_image)

        return stylized_image