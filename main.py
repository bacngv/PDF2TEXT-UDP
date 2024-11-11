import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import numpy as np
import pandas as pd
import easyocr
from tqdm.auto import tqdm
import cv2
import os
from pdf2image import convert_from_path
from pdf_processing import process_pdf

if __name__ == "__main__":
    pdf_path = r"C:\Users\Admin\Documents\HK1 24-25\Du_an_cong_nghe\PDF2TEXT-UDP\input\financial_statements.pdf"  # Path to your PDF
    output_folder = r"C:\Users\Admin\Documents\HK1 24-25\Du_an_cong_nghe\PDF2TEXT-UDP\output"  # Output folder
    process_pdf(pdf_path, output_folder)
