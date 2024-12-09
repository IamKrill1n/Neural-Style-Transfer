from nst_torch import trainer

model = trainer.FastStyleTransferTrainer()
model.train(style_image_path='style_images\\tsunami.jpg', content_image_path='content_images', epochs=2, batch_size=4)