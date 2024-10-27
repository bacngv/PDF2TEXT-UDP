import os
import pandas as pd

def save_remaining_text_to_txt(text, output_folder, page_num):
    txt_filename = os.path.join(output_folder, f'page_{page_num + 1}_remaining_text.txt')
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(text)

def save_to_csv(data, csv_filename):
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
