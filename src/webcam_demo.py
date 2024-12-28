import cv2
import torch
from torchvision import transforms
from PIL import Image
from nst_torch.inference import FastStyleTransfer
from nst_torch.config import device

# Load the style transfer model
model_path = 'models/mosaic.model'
style_transfer = FastStyleTransfer(model_path, device=device)

# Define the size of the frames
height, width = 480, 640
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image
    transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB
    transforms.Resize((height, width)),  # Resize to a fixed size
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_tensor = transform(frame).unsqueeze(0).to(device)
    stylized_frame_tensor = style_transfer.stylize(frame_tensor, return_tensor=True)
    stylized_frame = stylized_frame_tensor.squeeze(0).cpu().numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0)  # Convert from CHW to HWC
    stylized_frame = (stylized_frame * 255).astype('uint8')  # Convert to uint8
    stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Stylized Frame', stylized_frame)

    # Break the loop on 'q' key press or window close event
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Stylized Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()