import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw

def visualize_detected_tables(img, det_tables):
    plt.imshow(img)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']
        rect = patches.Rectangle(
            bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def visualize_cropped_table(cropped_table, cells):
    draw = ImageDraw.Draw(cropped_table)
    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    plt.imshow(cropped_table)
    plt.axis('off')
    plt.show()
