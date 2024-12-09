import torch
from .transformer_net import TransformerNet
from .config import device
from .utils import image_unloader, preserve_color_lab, preserve_color_ycbcr

class FastStyleTransfer:
    def __init__(self, model_path, device=device):
        self.device = torch.device(device)
        self.transformer = TransformerNet().to(self.device)
        self.transformer.load_state_dict(torch.load(model_path, map_location=self.device))
        self.transformer.eval()
        # training_state = torch.load(model_path, map_location=self.device)
        # state_dict = training_state["state_dict"]
        # self.transformer.load_state_dict(state_dict, strict=True)
        # self.transformer.eval()

    def stylize(self, content_image, preserve_color = False, return_tensor=True):
        with torch.no_grad():
            content_image = content_image.to(self.device)
            stylized_image = self.transformer(content_image)
            stylized_image.data.clamp_(0, 1)

        if preserve_color:
            stylized_image  = preserve_color_lab(content_image, stylized_image)

        if not return_tensor:
            stylized_image = image_unloader(stylized_image)

        return stylized_image