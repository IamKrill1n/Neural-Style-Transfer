# Neural-Style-Transfer

## Original NST
You can check out the file `src/original_style_transfer.ipynb` for example usage or use the Python script `src/original_style_transfer.py`:

```bash
python src/original_style_transfer.py --content_img_path <path_to_content_image> --style_img_path <path_to_style_image> --output_img_path <path_to_output_image>
```

## Fast NST 
You can checkout the file src\fast_style_transfer.ipynb for example usage for both training and inference
Or 
### Training the Model
Download data
You can use any images as content, I use the content images in the wikiart dataset in this link
```bash
https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000
```

Put the image files in the following folder: 

data/wikiart/

 use the following command:

GPU:
```bash
python src/train_script.py train --dataset data --save-model-dir model --checkpoint-model-dir checkpoints --cuda 1
```

CPU:
```bash
python src/train_script.py train --dataset data --save-model-dir model --checkpoint-model-dir checkpoints --cuda 0
```

### Inference
