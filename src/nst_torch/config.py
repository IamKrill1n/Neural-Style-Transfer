import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CNN = 'vgg19'
DEFAULT_IMSIZE = 512 if torch.cuda.is_available() else 128
DEFAULT_CONTENT_LAYERS = ['conv4_1']
DEFAULT_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
# DEFAULT_CONTENT_LAYERS = ['relu4_1']
# DEFAULT_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
DEFAULT_CONTENT_WEIGHTS = [1]
DEFAULT_STYLE_WEIGHTS = [1, 1, 1, 1, 1]
DEFAULT_INITIALIZATION = 'content'
DEFAULT_NUM_STEPS = 300
DEFAULT_ALPHA = 1
DEFAULT_BETA = 1e7
DEFAULT_TV_WEIGHT = 0
DEFAULT_OPTIMIZER = 'lbfgs'
DEFAULT_PRESERVE_COLOR = False
DEFAULT_RETURN_TENSOR = False
vgg16_layers_mapping = {
    'conv1_1': 0,
    'relu1_1': 1,
    'conv1_2': 2,
    'relu1_2': 3,
    'pool1': 4,
    'conv2_1': 5,
    'relu2_1': 6,
    'conv2_2': 7,
    'relu2_2': 8,
    'pool2': 9,
    'conv3_1': 10,
    'relu3_1': 11,
    'conv3_2': 12,
    'relu3_2': 13,
    'conv3_3': 14,
    'relu3_3': 15,
    'pool3': 16,
    'conv4_1': 17,
    'relu4_1': 18,
    'conv4_2': 19,
    'relu4_2': 20,
    'conv4_3': 21,
    'relu4_3': 22,
    'pool4': 23,
    'conv5_1': 24,
    'relu5_1': 25,
    'conv5_2': 26,
    'relu5_2': 27,
    'conv5_3': 28,
    'relu5_3': 29,
    'pool5': 30
}

vgg19_layers_mapping = {
    'conv1_1': 0,
    'relu1_1': 1,
    'conv1_2': 2,
    'relu1_2': 3,
    'pool1': 4,
    'conv2_1': 5,
    'relu2_1': 6,
    'conv2_2': 7,
    'relu2_2': 8,
    'pool2': 9,
    'conv3_1': 10,
    'relu3_1': 11,
    'conv3_2': 12,
    'relu3_2': 13,
    'conv3_3': 14,
    'relu3_3': 15,
    'conv3_4': 16,
    'relu3_4': 17,
    'pool3': 18,
    'conv4_1': 19,
    'relu4_1': 20,
    'conv4_2': 21,
    'relu4_2': 22,
    'conv4_3': 23,
    'relu4_3': 24,
    'conv4_4': 25,
    'relu4_4': 26,
    'pool4': 27,
    'conv5_1': 28,
    'relu5_1': 29,
    'conv5_2': 30,
    'relu5_2': 31,
    'conv5_3': 32,
    'relu5_3': 33,
    'conv5_4': 34,
    'relu5_4': 35,
    'pool5': 36
}