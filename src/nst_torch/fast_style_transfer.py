import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from .config import *
from .loss import gram_matrix, normalize_batch
from .utils import image_loader, image_unloader
from .transformer_net import TransformerNet
from .perceptual_loss_net import Vgg16, Vgg19

class FastStyleTransfer:
    def __init__(self, cnn, style_image_path, device=device):
        self.device = device
        self.transformer_net = TransformerNet().to(self.device)
        if cnn == 'vgg16':
            self.cnn = Vgg16().to(self.device)
        elif cnn == 'vgg19':
            self.cnn = Vgg19().to(self.device)
        else:
            raise ValueError("Only VGG16 and VGG19 are supported")
        self.style_image = image_loader(style_image_path, imsize=self.image_size).to(self.device)
        self.gram_style = self._get_style_grams(self.style_image)

    def _get_style_grams(self, style_image):
        style_image = normalize_batch(style_image)
        features_style = self.vgg(style_image)
        gram_style = [gram_matrix(y) for y in features_style]
        return gram_style

    def train(self, dataset_path, epochs=2, batch_size=4, learning_rate=1e-3, 
              log_interval=500, checkpoint_dir=None):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        train_dataset = datasets.ImageFolder(dataset_path, transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        self.transformer.train()
        optimizer = optim.Adam(self.transformer.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        for e in range(epochs):
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(self.device)
                y = self.transformer(x)

                y = normalize_batch(y)
                x = normalize_batch(x)

                features_y = self.vgg(y)
                features_x = self.vgg(x)

                content_loss = self.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2.detach())

                style_loss = 0.
                for ft_y, gm_s in zip(features_y, self.gram_style):
                    gm_y = gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s.repeat(n_batch, 1, 1))
                style_loss *= self.style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                if (batch_id + 1) % log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_loader.dataset),
                        agg_content_loss / (batch_id + 1),
                        agg_style_loss / (batch_id + 1),
                        (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    print(mesg)

            if checkpoint_dir is not None:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                transformer_cpu = self.transformer.to('cpu').eval()
                ckpt_model_filename = f"ckpt_epoch_{e}.pth"
                ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
                torch.save(transformer_cpu.state_dict(), ckpt_model_path)
                self.transformer.to(self.device).train()

    def stylize(self, content_image_path, output_image_path=None):
        content_image = image_loader(content_image_path, imsize=self.image_size).to(self.device)
        with torch.no_grad():
            self.transformer.eval()
            output = self.transformer(content_image).cpu()
        output_image = image_unloader(output.squeeze(0))
        if output_image_path:
            output_image.save(output_image_path)
        return output_image