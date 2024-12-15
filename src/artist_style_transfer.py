import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Style Transfer Parameter Setup")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing content images.')
    parser.add_argument('--style_name', type=str, default='ukiyoe', help='Name of the style to be applied.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the output images will be saved.')

    return parser.parse_args()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=7, strides=1, padding=0, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.blocks = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
                                        nn.InstanceNorm2d(num_features=filters),
                                        activation(inplace=True))

    def forward(self, input_tensor):
        x = self.blocks(input_tensor)
        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=7, strides=1, padding=0, output_padding=1, activation=nn.ReLU):
        super(DeConvBlock, self).__init__()
        self.blocks = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels=filters, kernel_size=kernel_size,
                                                       stride=strides, padding=padding, output_padding=output_padding),
                                    nn.InstanceNorm2d(num_features=filters),
                                    activation(inplace=True))

    def forward(self, input_tensor):
        x = self.blocks(input_tensor)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, padding=0, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.conv_blocks = nn.Sequential(
                                            nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
                                            nn.ReflectionPad2d(1),
                                            nn.InstanceNorm2d(num_features=filters),
                                            activation(inplace=True),
                                            nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
                                            nn.ReflectionPad2d(1),
                                            nn.InstanceNorm2d(num_features=filters))

    def forward(self, input_tensor):
        x = self.conv_blocks(input_tensor)
        x = x + input_tensor
        return x

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None
    def forward(self, x):
        pass
    def summary(self):
        if self.model != None:
            print('=================================================================')
            print(self.model)
            print('=================================================================')
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('Total params: {:,}'.format(total_params))
            print('Trainable params: {:,}'.format(trainable_params))
            print('Non-trainable params: {:,}'.format(total_params - trainable_params))
        else:
            print('Model not created')

class ResNetGenerator(BaseModel):
    def __init__(self, input_channel=3, output_channel=3, filters=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        # Downsample layers
        layers = [
                      nn.ReflectionPad2d(3),
                      ConvBlock(in_channels=input_channel, filters=filters, kernel_size=7, strides=1, activation=nn.LeakyReLU),
                      ConvBlock(in_channels=filters, filters=filters * 2, kernel_size=3, strides=2, padding=1, activation=nn.LeakyReLU),
                      ConvBlock(in_channels=filters * 2, filters=filters * 4, kernel_size=3, strides=2, padding=1, activation=nn.LeakyReLU)
        ]

        # Residual layers
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_channels=filters * 4, filters=filters * 4, kernel_size=3, strides=1, activation=nn.LeakyReLU))

        # Upsample layers
        layers += [
                        DeConvBlock(in_channels=filters * 4, filters=filters * 2, kernel_size=3, strides=2, padding=1, output_padding=1, activation=nn.LeakyReLU),
                        DeConvBlock(in_channels=filters * 2, filters=filters, kernel_size=3, strides=2, padding=1, output_padding=1, activation=nn.LeakyReLU),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size=7, stride=1, padding=0)
        ]
        # Output layer
        layers += [nn.Tanh()]

        # Create model
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.model(input_tensor)
        return x

class PatchGANDiscriminator(BaseModel):
    def __init__(self, input_channel, filters=64):
        super(PatchGANDiscriminator, self).__init__()

        layers = [
                      nn.Conv2d(in_channels=input_channel, out_channels=filters, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(inplace=True),
                      ConvBlock(in_channels=filters, filters=filters * 2, kernel_size=4, strides=2, padding=1, activation=nn.LeakyReLU),
                      ConvBlock(in_channels=filters * 2, filters=filters * 4, kernel_size=4, strides=2, padding=1, activation=nn.LeakyReLU),
                      ConvBlock(in_channels=filters * 4, filters=filters * 8, kernel_size=4, strides=1, padding=1, activation=nn.LeakyReLU),
        ]
        # Output layer
        layers += [nn.Conv2d(in_channels=filters * 8, out_channels=1, kernel_size=4, stride=1, padding=1)]
        # Create model
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.model(input_tensor)
        return x
    
ngpu = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_XtoY = ResNetGenerator(input_channel=3, output_channel=3, filters=64, n_blocks=9).to(device)
G_YtoX = ResNetGenerator(input_channel=3, output_channel=3, filters=64, n_blocks=9).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    G_XtoY = nn.DataParallel(G_XtoY, list(range(ngpu)))
    G_YtoX = nn.DataParallel(G_YtoX, list(range(ngpu)))

Dx = PatchGANDiscriminator(input_channel=3, filters=64).to(device)
Dy = PatchGANDiscriminator(input_channel=3, filters=64).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    Dx = nn.DataParallel(Dx, list(range(ngpu)))
    Dy = nn.DataParallel(Dy, list(range(ngpu)))

def denormalize(images, std=0.5, mean=0.5):
    images = (images * std) + mean
    return images

def deprocess(input_tensor):
    if len(input_tensor.shape) == 3:
        return np.transpose(denormalize(input_tensor.to(device).cpu()), (1,2,0))
    elif len(input_tensor.shape) == 4:
        return np.transpose(denormalize(input_tensor.to(device).cpu()), (0, 2,3,1))
    
class GeneratorDataset(data.Dataset):
    """Load images first for generator. """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        sample = Image.open(img_path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)

        return sample

preprocess_test_transformations = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

def save_test_images(model1, model2, test_input, step=0, save=False, show_result=False):
    '''
        Generate images and cycled images, then save them as jpg
    '''

    with torch.no_grad():
        prediction1 = model1(test_input.to(device))
        prediction2 = model2(prediction1)
    test_input = test_input.cpu()
    prediction1 = prediction1.cpu()
    prediction2 = prediction2.cpu()

    display_list = [test_input, prediction1, prediction2]
    figure_title = ['Input Image', 'Predicted Image', 'Cycled Image']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save:
        img = denormalize(prediction1)[0]
        T = transforms.ToPILImage()
        img = T(img)
        img.save(os.path.join(output_dir, '{}.jpg'.format(step)))

    if show_result:
        plt.figure(figsize=(12, 12))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(figure_title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(deprocess(display_list[i])[0])
            plt.axis('off')
        plt.show()

if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    style_name = args.style_name
    output_dir = args.output_dir
    checkpoint_path = os.path.join("./models", f'PAPM_{style_name}.model')

    if not os.path.exists(checkpoint_path):
        print('Checkpoint path doesn\'t exist! Please download the available model ')

    checkpoint = torch.load(checkpoint_path)
    G_XtoY.load_state_dict(checkpoint['G_XtoY'])
    G_YtoX.load_state_dict(checkpoint['G_YtoX'])
    Dx.load_state_dict(checkpoint['Dx'])
    Dy.load_state_dict(checkpoint['Dy'])
    CURRENT_EPOCH = checkpoint['epoch']
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))


    test_data = iter(torch.utils.data.DataLoader(GeneratorDataset(root_dir=input_dir, transform=preprocess_test_transformations), batch_size=1, shuffle=False, num_workers=0))
    for step, image in enumerate(tqdm(test_data)):
        save_test_images(G_YtoX, G_XtoY, image, step=step, save=True, show_result=False)
