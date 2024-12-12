# Neural-Style-Transfer

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Neural-Style-Transfer.git
    cd Neural-Style-Transfer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Original NST
You can check out the file `src/original_style_transfer.ipynb` for example usage or use the Python script `src/original_style_transfer.py`:

```bash
python src/original_style_transfer.py --content_img_path <path_to_content_image> --style_img_path <path_to_style_image> --output_img_path <path_to_output_image>
```

### Fast NST 
#### Inference
To perform style transfer on an image using a pre-trained model:
```bash
python src/infer_script.py --content_img_path <path_to_content_image> --style <style_name> --output_img_path <path_to_output_image>
```


#### Training the Model

To train the fast NST model, follow these steps:

1. **Download the dataset**: You can use any images as content. For example, you can use the content images in the WikiArt dataset available [here](https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000?select=style)

```
data/
└── wikiart/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```
2. **Create the necessary directories** for saving models and checkpoints:

```bash
mkdir models checkpoints
```

Train on GPU:
```bash
python src/train_script.py train --dataset data --save-model-dir models --checkpoint-model-dir checkpoints --cuda 1
```

Train on CPU (not advised):
```bash
python src/train_script.py train --dataset data --save-model-dir model --checkpoint-model-dir checkpoints --cuda 0
```

For more args options checkout the source code
