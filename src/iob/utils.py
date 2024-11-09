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

def preserve_content_color(content_img, output_img):
    # Convert images to PIL format
    content = unloader(content_img.cpu().clone().squeeze(0))
    output = unloader(output_img.cpu().clone().squeeze(0))

    # Convert images to LAB color space
    content_lab = content.convert('LAB')
    output_lab = output.convert('LAB')

    # Split channels
    l_content, a_content, b_content = content_lab.split()
    l_output, _, _ = output_lab.split()

    # Merge L channel from output with A and B channels from content
    result_lab = Image.merge('LAB', (l_output, a_content, b_content))

    # Convert back to RGB
    result = result_lab.convert('RGB')

    # Convert back to tensor
    loader = transforms.Compose([
        transforms.ToTensor()
    ])
    result = loader(result).unsqueeze(0).to(device)
    return result

def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')