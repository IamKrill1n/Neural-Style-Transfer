import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from .config import *

unloader = transforms.ToPILImage()

def image_loader(image_path, imsize=DEFAULT_IMSIZE):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)