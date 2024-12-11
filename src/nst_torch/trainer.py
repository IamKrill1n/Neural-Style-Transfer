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

    def get_optimizer(self, optimizer, learning_rate):
        if learning_rate is None:
            if self.optimizer == 'adam':
                learning_rate = 0.005
            elif self.optimizer == 'sgd':
                learning_rate = 0.005

        if optimizer == 'adam':
            return optim.Adam(self.transformer.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            return optim.SGD(self.transformer.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}") 

    def train(self, style_image_path, content_image_path, imsize = DEFAULT_TRAIN_IMSIZE, 
            alpha = DEFAULT_ALPHA, 
            beta = DEFAULT_BETA, 
            tv_weight = DEFAULT_TV_WEIGHT, 
            epochs=2, 
            batch_size=4, 
            learning_rate=None, 
            checkpoint_path=None,
            model_path=None,
            log_interval=500,
            checkpoint_interval=2000):
        
        self.transformer.train()
        # Get style features and gram matrixes
        style_image = image_loader(style_image_path, imsize=imsize).to(self.device)
        style = style_image.repeat(batch_size, 1, 1, 1).to(device)
        features_style = self.cnn(style)
        gram_style = [gram_matrix(features_style[layer].detach()) for layer in self.style_layers]

        # Get content loader  
        train_loader = get_content_loader(content_image_path, imsize, batch_size)
        
        # Get optmizier
        optimizer = self.get_optimizer(self.optimizer, learning_rate)
        print("Training starts...")
        for e in range(epochs):
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(device)
                y = self.transformer(x)

                features_y = self.cnn(y)
                features_x = self.cnn(x)

                content_score = 0.
                for layer, weight in zip(self.content_layers, self.content_weights):
                    content_score += weight * content_loss(features_y[layer], features_x[layer].detach())

                style_score = 0.
                for layer, weight, gm_s in zip(self.style_layers, self.style_weights, gram_style):
                    gm_y = gram_matrix(features_y[layer])
                    style_score += weight * content_loss(gm_y, gm_s[:n_batch, :, :])

                total_loss = alpha * content_score + beta * style_score + tv_weight * total_variation_loss(y)
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_score.item()
                agg_style_loss += style_score.item()

                if (batch_id + 1) % log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_loader.dataset),
                                    agg_content_loss / (batch_id + 1),
                                    agg_style_loss / (batch_id + 1),
                                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    print(mesg)

                if checkpoint_path is not None and (batch_id + 1) % checkpoint_interval == 0:
                    self.transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                    ckpt_model_path = os.path.join(checkpoint_path, ckpt_model_filename)
                    torch.save(self.transformer.state_dict(), ckpt_model_path)
                    self.transformer.to(device).train()

        # save model
        self.transformer.eval().cpu()
        save_model_filename = "epoch_" + str(epochs) + ".model"
        save_model_path = os.path.join(model_path, save_model_filename)
        torch.save(self.transformer.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)

        

    def stylize(self, content_image, preserve_color = False, return_tensor=True):
        self.transformer.to(self.device).eval()
        with torch.no_grad():
            content_image = content_image.to(self.device)
            stylized_image = self.transformer(content_image)
            if stylized_image.max() > 100:
                stylized_image = stylized_image/ 255.0

        if preserve_color:
            stylized_image  = preserve_color_lab(content_image, stylized_image)

        if not return_tensor:
            stylized_image = image_unloader(stylized_image)

        return stylized_image