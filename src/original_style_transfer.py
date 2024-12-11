import argparse
from nst_torch.style_transfer import StyleTransfer
from nst_torch.config import (
    DEFAULT_CNN,
    DEFAULT_CONTENT_LAYERS,
    DEFAULT_STYLE_LAYERS,
    DEFAULT_CONTENT_WEIGHTS,
    DEFAULT_STYLE_WEIGHTS,
    DEFAULT_OPTIMIZER,
    DEFAULT_IMSIZE,
    DEFAULT_INITIALIZATION,
    DEFAULT_NUM_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_TV_WEIGHT,
    DEFAULT_PRESERVE_COLOR,
    DEFAULT_RETURN_TENSOR
)

def main():
    parser = argparse.ArgumentParser(description='Style Transfer Script')
    parser.add_argument('--content_img_path', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style_img_path', type=str, required=True, help='Path to the style image')
    parser.add_argument('--imsize', type=int, default=DEFAULT_IMSIZE, help='Size of the output image')
    parser.add_argument('--initialization', type=str, choices=['content', 'random'], default=DEFAULT_INITIALIZATION, help='Initialization method')
    parser.add_argument('--num_steps', type=int, default=DEFAULT_NUM_STEPS, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate of the optimizer')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Content weight')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA, help='Style weight')
    parser.add_argument('--tv_weight', type=float, default=DEFAULT_TV_WEIGHT, help='Total variation weight')
    parser.add_argument('--preserve_color', action='store_true', default=DEFAULT_PRESERVE_COLOR, help='Preserve color of the content image')
    parser.add_argument('--return_tensor', action='store_true', default=DEFAULT_RETURN_TENSOR, help='Return tensor instead of image')
    parser.add_argument('--output_img_path', type=str, required=True, help='Output image path')
    # Arguments for StyleTransfer initialization
    parser.add_argument('--cnn', type=str, choices=['vgg16', 'vgg19'], default=DEFAULT_CNN, help='CNN model to use')
    parser.add_argument('--content_layers', nargs='+', default=DEFAULT_CONTENT_LAYERS, help='Content layers')
    parser.add_argument('--style_layers', nargs='+', default=DEFAULT_STYLE_LAYERS, help='Style layers')
    parser.add_argument('--content_weights', nargs='+', type=float, default=DEFAULT_CONTENT_WEIGHTS, help='Content weights')
    parser.add_argument('--style_weights', nargs='+', type=float, default=DEFAULT_STYLE_WEIGHTS, help='Style weights')
    parser.add_argument('--optimizer', type=str, choices=['lbfgs', 'adam', 'sgd'], default=DEFAULT_OPTIMIZER, help='Optimizer to use')

    args = parser.parse_args()

    st = StyleTransfer(
        cnn=args.cnn,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        content_weights=args.content_weights,
        style_weights=args.style_weights,
        optimizer=args.optimizer
    )
    output = st.stylize(
        content_img_path=args.content_img_path,
        style_img_path=args.style_img_path,
        imsize=args.imsize,
        initialization=args.initialization,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        tv_weight=args.tv_weight,
        preserve_color=args.preserve_color,
        return_tensor=args.return_tensor
    )

    if not args.return_tensor:
        output.save(args.output_img_path)
    else:
        print("Output is a tensor. Cannot save as image.")

if __name__ == '__main__':
    main()