import torch
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .config import *
import os

unloader = transforms.ToPILImage()

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def image_loader(image_path, imsize=DEFAULT_IMSIZE, device = device, scale=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    if imsize == None or imsize == -1:
        imsize = image.size[1]

    if isinstance(imsize, int):
        width, height = image.size
        width *= imsize / height
        imsize = (imsize, int(width))

    if not scale:
        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor(),
        ])
    else:
        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def image_unloader(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    return image

def imshow(tensor, title=None):
    image = image_unloader(tensor)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

def get_content_loader(content_dir, imsize=DEFAULT_IMSIZE, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(content_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

def preserve_color_lab(content_tensor, output_tensor):
    # Input 2 tensors and return a tensor

    # Convert images to PIL format
    content = unloader(content_tensor.cpu().clone().squeeze(0))
    output = unloader(output_tensor.cpu().clone().squeeze(0))

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

def preserve_color_ycbcr(content_tensor, output_tensor):
    # Input 2 tensors and return a tensor
    content = unloader(content_tensor.cpu().clone().squeeze(0))
    output = unloader(output_tensor.cpu().clone().squeeze(0))
    content_channels = list(content.convert('YCbCr').split())
    output_channels = list(output.convert('YCbCr').split())
    content_channels[0] = output_channels[0]
    result_ycbcr = Image.merge('YCbCr', content_channels).convert('RGB')
    loader = transforms.Compose([
        transforms.ToTensor()
    ])
    result = loader(result_ycbcr).unsqueeze(0).to(device)
    return result

# if __name__ == '__main__':
#     content_dir = 'wikiart/'
#     loader = get_content_loader(content_dir, batch_size=4, shuffle=True, imsize=DEFAULT_IMSIZE, device=device)
#     for batch_id, content_batch in enumerate(loader):
#         print(f"Batch {batch_id}: {content_batch.shape}")
#         imshow(content_batch[0], title='Content Image')
#         if batch_id == 0:
#             break
#     plt.show()