import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMSIZE = (512, 512) if torch.cuda.is_available() else (128, 128)
DEFAULT_CONTENT_LAYERS = ['conv_4']
DEFAULT_STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# DEFAULT_CONTENT_LAYERS = ['relu_4']
# DEFAULT_STYLE_LAYERS = ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']
DEFAULT_CONTENT_WEIGHTS = [1]
DEFAULT_STYLE_WEIGHTS = [1, 1, 1, 1, 1]
DEFAULT_INITIALIZATION = 'content'
DEFAULT_NUM_STEPS = 300
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1e5
DEFAULT_OPTIMIZER = 'lbfgs'
DEFAULT_PRESERVE_COLOR = False
DEFAULT_RETURN_TENSOR = False