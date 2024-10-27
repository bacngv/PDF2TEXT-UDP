import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection

def load_detection_model():
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

def load_structure_model(device):
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )
    structure_model.to(device)
    return structure_model

def detect_objects(model, pixel_values, device):
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': bbox})
    return objects
