import easyocr
import numpy as np
from PIL import Image

def apply_ocr_remaining_area(remaining_area_image):
    reader = easyocr.Reader(['vi'])
    text = reader.readtext(np.array(remaining_area_image), detail=0)
    return ' '.join(text)

def apply_ocr(cell_coordinates, cropped_table):
    reader = easyocr.Reader(['vi'])
    extracted_data = []

    for row in cell_coordinates:
        row_data = []
        for cell in row:
            x_min, y_min, x_max, y_max = cell['bbox']
            cell_image = cropped_table.crop((x_min, y_min, x_max, y_max))
            text = reader.readtext(np.array(cell_image), detail=0)
            row_data.append(' '.join(text))
        extracted_data.append(row_data)

    return extracted_data
