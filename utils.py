from PIL import Image
import torch
from torchvision import transforms
import numpy as np

class MaxResize:
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        scale = self.max_size / max(width, height)
        return image.resize((int(scale * width), int(scale * height)))
# Load and preprocess the image
def load_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return image
def preprocess_image(image, max_size=800):
    detection_transform = transforms.Compose([
        MaxResize(max_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = detection_transform(image).unsqueeze(0)
    return pixel_values

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def remove_table_areas(image, detected_bboxes):
    img_arr = np.array(image)
    for bbox in detected_bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        img_arr[y_min:y_max, x_min:x_max] = 255  # Set the area to white
    return Image.fromarray(img_arr)
