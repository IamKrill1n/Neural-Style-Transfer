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

### Artist Model (PAPM method)
You first need to download one of our pretrained model (for example 'PAPM_ukiyoe.model') from [this link here](https://husteduvn-my.sharepoint.com/:f:/g/personal/binh_nd225475_sis_hust_edu_vn/EiRV0jQp-ZhMoFTtUgXzjYABrcelOx1l4W2VEOtwpewF1Q?e=6BMUvk) and put it in the 'models' directory

After that, you can use the following command to perform inference on all images from your directory:
```bash
python src/artist_style_transfer.py --input_dir <your-input-directory> --output_dir <your-output-directory> --style_name <style-name>
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

#### Webcam Inference

You will need to install open-cv for this
```bash
pip install opencv-python
```

Then run the following command:
```bash
python src/webcam_demo.py
```

## Reference

- [Evaluation in Neural Style Transfer: A Review](https://doi.org/10.48550/arXiv.2401.17109) by Eleftherios Ioannou and Steve Maddock (2024).

- [A neural algorithm of artistic style](https://doi.org/10.1167/16.12.326) by Gatys, L., Ecker, A., & Bethge, M. (2016). Journal of Vision, 16(12), 326.

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://doi.org/10.1007/978-3-319-46475-6_43) by Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Lecture notes in computer science, pp. 694–711.

- [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://doi.org/10.1109/iccv.2017.244) by Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). ICCV 2017, pp. 2242–2251.

- [Neural Style Transfer: A Review](https://doi.org/10.1109/tvcg.2019.2921336) by Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). IEEE Transactions on Visualization and Computer Graphics, 26(11), 3365–3385.

- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://doi.org/10.48550/arxiv.1607.08022) by Ulyanov, D., Vedaldi, A., & Lempitsky, V. S. (2016).

- [Multi-head Mutual-attention CycleGAN for Unpaired Image-to-Image Translation](https://doi.org/10.1049/iet-ipr.2019.1153) by Ji, Wei, Guo, Jing, & Li, Yun. (2020). IET Image Processing.

Code we found useful: 

- [Pytorch example code](https://github.com/pytorch/examples/tree/main/fast_neural_style)
  
- [CycleGAN notebook](https://github.com/henry32144/cyclegan-notebook.git)
