import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .config import *
import os

unloader = transforms.ToPILImage()

def image_loader(image_path, imsize=DEFAULT_IMSIZE, device = device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    if isinstance(imsize, int):
        width, height = image.size
        width *= imsize / height
        imsize = (imsize, int(width))

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
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

class ContentImageDataset(Dataset):
    def __init__(self, content_dir, imsize=DEFAULT_IMSIZE):
        self.content_dir = content_dir
        self.image_files = [
            os.path.join(content_dir, fname)
            for fname in os.listdir(content_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.png'))
        ]
        self.imsize = imsize

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = image_loader(img_path, self.imsize, device = 'cpu').squeeze(0)
        return image

def get_content_loader(content_dir, imsize=DEFAULT_IMSIZE, batch_size=4, shuffle=True):
    dataset = ContentImageDataset(content_dir, imsize=imsize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

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