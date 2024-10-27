import os
from pdf2image import convert_from_path
from utils import preprocess_image, remove_table_areas, MaxResize 
from models import load_detection_model, load_structure_model, detect_objects, outputs_to_objects
from visualization import visualize_detected_tables, visualize_cropped_table
from ocr import apply_ocr, apply_ocr_remaining_area
from io_utils import save_remaining_text_to_txt, save_to_csv
from torchvision import transforms 

# Main function to process the PDF file
def process_pdf(pdf_path, output_folder):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    model, device = load_detection_model()
    structure_model = load_structure_model(device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num, image in enumerate(images):
        print(f"Processing page {page_num + 1}/{len(images)}")

        # Detect tables in the image
        pixel_values = preprocess_image(image)
        outputs = detect_objects(model, pixel_values, device)

        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"
        objects = outputs_to_objects(outputs, image.size, id2label)

        # Visualize detected tables
        visualize_detected_tables(image, objects)

        # Create a mask for the detected table bounding boxes
        detected_bboxes = [obj['bbox'] for obj in objects if obj['label'] in ['table', 'table rotated']]

        # Create a new image with the table areas removed
        image_without_tables = remove_table_areas(image, detected_bboxes)

        # Perform OCR on the remaining area of the image
        remaining_text = apply_ocr_remaining_area(image_without_tables)
        save_remaining_text_to_txt(remaining_text, output_folder, page_num)

        # Process each cropped table and save data as CSV
        table_data_list = []
        for idx, bbox in enumerate(detected_bboxes):
            x_min, y_min, x_max, y_max = bbox
            cropped_table = image.crop((x_min, y_min, x_max, y_max))
            table_data = process_cropped_table(cropped_table, structure_model, device, page_num, idx, output_folder)
            table_data_list.append(table_data)

# Function to process each cropped table and save to CSV
def process_cropped_table(cropped_table, structure_model, device, page_num, table_index, output_folder):
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = structure_transform(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # Update id2label to include "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)

    # Visualize the detected cells in the cropped table
    visualize_cropped_table(cropped_table, cells)

    # Apply OCR to the detected cells
    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates, cropped_table)

    # Save extracted data to CSV
    csv_filename = os.path.join(output_folder, f'page_{page_num + 1}_table_{table_index + 1}.csv')
    save_to_csv(data, csv_filename)

    return data

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'bbox': cell_bbox})
        cell_coordinates.append(row_cells)

    return cell_coordinates
