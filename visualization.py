import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw

# Visualize the detected tables
def visualize_detected_tables(img, det_tables):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']
        facecolor = (1, 0, 0.45) if det_table['label'] == 'table' else (0.95, 0.6, 0.1)
        edgecolor = facecolor
        alpha = 0.3
        linewidth = 2
        hatch='//////'

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.show()

# Visualize the cropped table with detected cells
def visualize_cropped_table(cropped_table, cells):
    cropped_table_visualized = cropped_table.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    plt.imshow(cropped_table_visualized)
    plt.axis('off')
    plt.show()
