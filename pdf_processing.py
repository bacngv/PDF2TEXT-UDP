import os
from pdf2image import convert_from_path
from utils import preprocess_image, remove_table_areas
from models import load_detection_model, load_structure_model, detect_objects, outputs_to_objects
from visualization import visualize_detected_tables, visualize_cropped_table
from ocr import apply_ocr, apply_ocr_remaining_area
from io_utils import save_remaining_text_to_txt, save_to_csv

def process_pdf(pdf_path, output_folder):
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

        objects = outputs_to_objects(outputs, image.size, model.config.id2label)
        visualize_detected_tables(image, objects)

        # Remove detected tables from the image
        detected_bboxes = [obj['bbox'] for obj in objects if obj['label'] in ['table', 'table rotated']]
        image_without_tables = remove_table_areas(image, detected_bboxes)

        # Apply OCR on the remaining area
        remaining_text = apply_ocr_remaining_area(image_without_tables)
        save_remaining_text_to_txt(remaining_text, output_folder, page_num)

        # Process and save tables as CSV
        for idx, bbox in enumerate(detected_bboxes):
            process_cropped_table(image.crop(bbox), structure_model, device, page_num, idx, output_folder)

def process_cropped_table(cropped_table, structure_model, device, page_num, table_index, output_folder):
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = structure_transform(cropped_table).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # Update id2label to include "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
    visualize_cropped_table(cropped_table, cells)

    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates, cropped_table)

    csv_filename = os.path.join(output_folder, f'page_{page_num + 1}_table_{table_index + 1}.csv')
    save_to_csv(data, csv_filename)

def get_cell_coordinates_by_row(table_data):
    """Extract cell coordinates by organizing rows and columns."""
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])  # Sort rows by their Y-coordinate (top-to-bottom)
    columns.sort(key=lambda x: x['bbox'][0])  # Sort columns by their X-coordinate (left-to-right)

    def find_cell_coordinates(row, column):
        """Find the cell bounding box based on row and column coordinates."""
        return [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'bbox': cell_bbox})
        cell_coordinates.append(row_cells)

    return cell_coordinates
