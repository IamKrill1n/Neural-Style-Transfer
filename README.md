# Neural-Style-Transfer

## Original NST



## Fast NST 
### Training the Model
Download data
You can use any images as content, I use the content images in the wikiart dataset in this link
```bash
https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000
```

Put the image files in the following folder: 

data/wikiart/

To train the model, use the following command:

GPU:
```bash
python src/train_script.py train --dataset data --save-model-dir model --checkpoint-model-dir checkpoints --cuda 1
```

CPU:
```bash
python src/train_script.py train --dataset data --save-model-dir model --checkpoint-model-dir checkpoints --cuda 0
```

### Inference

Currently: fast_style_transfer.ipynb