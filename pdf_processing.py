import os
from pdf2image import convert_from_path
from utils import preprocess_image, remove_table_areas, MaxResize 
from models import load_detection_model, load_structure_model, detect_objects, outputs_to_objects
from visualization import visualize_detected_tables, visualize_cropped_table
from ocr import apply_ocr, apply_ocr_remaining_area
from io_utils import save_remaining_text_to_txt, save_to_csv
from torchvision import transforms
import torch
import cv2
import numpy as np
import RRDBNet_arch as arch 

def load_super_resolution_model(model_path='./PDF2TEXT-UDP/models/RRDB_ESRGAN_x4.pth', device=torch.device('cuda')):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

def apply_super_resolution(image, model, device):
    img = np.array(image)[:, :, ::-1]  
    img = img * 1.0 / 255  
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  
    output = (output * 255.0).round().astype(np.uint8)  
    return output

def process_pdf(pdf_path, output_folder):
    images = convert_from_path(pdf_path)


    model, device = load_detection_model()
    structure_model = load_structure_model(device)
    sr_model = load_super_resolution_model(device=device)  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num, image in enumerate(images):
        print(f"Processing page {page_num + 1}/{len(images)}")

        sr_image = apply_super_resolution(image, sr_model, device)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)  
        image = transforms.ToPILImage()(torch.tensor(sr_image).permute(2, 0, 1))  

        
        pixel_values = preprocess_image(image)
        outputs = detect_objects(model, pixel_values, device)

        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"
        objects = outputs_to_objects(outputs, image.size, id2label)

        visualize_detected_tables(image, objects)

        detected_bboxes = [obj['bbox'] for obj in objects if obj['label'] in ['table', 'table rotated']]

        image_without_tables = remove_table_areas(image, detected_bboxes)

        remaining_text = apply_ocr_remaining_area(image_without_tables)
        save_remaining_text_to_txt(remaining_text, output_folder, page_num)

        table_data_list = []
        for idx, bbox in enumerate(detected_bboxes):
            x_min, y_min, x_max, y_max = bbox
            cropped_table = image.crop((x_min, y_min, x_max, y_max))
            table_data = process_cropped_table(cropped_table, structure_model, device, page_num, idx, output_folder)
            table_data_list.append(table_data)

def process_cropped_table(cropped_table, structure_model, device, page_num, table_index, output_folder):
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = structure_transform(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = structure_model(pixel_values)
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
    visualize_cropped_table(cropped_table, cells)

    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates, cropped_table)

    csv_filename = os.path.join(output_folder, f'page_{page_num + 1}_table_{table_index + 1}.csv')
    save_to_csv(data, csv_filename)

    return data