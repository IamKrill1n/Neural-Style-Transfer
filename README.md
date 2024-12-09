# Neural-Style-Transfer

## To-do List

### Experiment with Hyperparameters in the Original Method and other Improvemnents
- Change the layers and layers' weights and observe the effects.
- Adjust the content/style weights (alpha/beta), may be try Patch-based Style loss.
- Modify the initialization, for example:
    - White noise
    - Content image
    - Blurred content image
    - Partially stylized image

### Explore Other Methods
- Read the survey and try other methods: [Neural Style Transfer: A Review](https://arxiv.org/pdf/1705.04058)
- Read the survey about evaluation: [Evaluation in Neural Style Transfer: A Review](https://arxiv.org/pdf/2401.17109)

## Fast NST 
### Training the Model

To train the model, use the following command:

```bash
python src/nst_torch/train_script.py train --dataset data --save-model-dir checkpoints --cuda 1
```

### Inference

Currently: fast_style_transfer.ipynb