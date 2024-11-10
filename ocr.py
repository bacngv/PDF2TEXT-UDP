import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
import RRDBNet_arch as arch

# Initialize the OCR reader
reader = easyocr.Reader(['vi'])

# Load the RRDBNet model for image enhancement
model_path = './PDF2TEXT-UDP/models/RRDB_ESRGAN_x4.pth'  # Ensure this path is correct
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

def enhance_image(image):
    """Enhances an image using RRDBNet."""
    img = np.array(image) * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)



def apply_ocr_remaining_area(remaining_area_image):
    enhanced_image = enhance_image(remaining_area_image)  # Enhance the image before OCR
    text = reader.readtext(np.array(enhanced_image), detail=0)
    return ' '.join(text)

def apply_ocr(cell_coordinates, cropped_table):
    extracted_data = []

    for row in cell_coordinates:
        row_data = []
        for cell in row:
            x_min, y_min, x_max, y_max = cell['bbox']
            cell_image = cropped_table.crop((x_min, y_min, x_max, y_max))
            enhanced_cell_image = enhance_image(cell_image)  # Enhance each cell image
            text = reader.readtext(np.array(enhanced_cell_image), detail=0)
            row_data.append(' '.join(text))
        extracted_data.append(row_data)

    return extracted_data
