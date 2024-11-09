import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
from utils import rescale_bboxes
# Load the model for object detection
def load_detection_model():
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

# Load the model for structure recognition
def load_structure_model(device):
    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)
    return structure_model
# Load and preprocess the image
def load_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return image

# Run detection on the image
def detect_objects(model, pixel_values, device):
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs

def outputs_to_objects(outputs, img_size, id2label, threshold=0.8):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if score >= threshold and not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})

    return objects

