from nst_torch.inference import FastStyleTransfer
import argparse
from nst_torch.utils import image_unloader, image_loader
import os

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--content_img_path', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, choices=['mosaic', 'starry_night', 'sketch', 'monet'], help='Trained style')
    parser.add_argument('--output_img_path', type=str, required=True, help='Path to output image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--preserve_color', type=bool, default = False, help='Preserve color of content image')
    args = parser.parse_args()



    model = FastStyleTransfer(model_path = 'models/' + args.style + '.model', device=args.device)
    content_img = image_loader(args.content_img_path, imsize = None, scale = True)
    stylized_img = model.stylize(content_img, preserve_color = False, return_tensor = True)
    stylized_img = image_unloader(stylized_img)
    stylized_img.save(os.path.join(args.output_img_path, 'output.png'))

if __name__ == '__main__':
    main()