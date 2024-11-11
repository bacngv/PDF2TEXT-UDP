import os
import pytesseract
from pdf2image import convert_from_path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import editdistance  # Thư viện để tính CER và WER

# Đường dẫn tới thư mục chứa PDF và output
input_folder = "C:/Users/Admin/Documents/HK1 24-25/Du_an_cong_nghe/PDF2TEXT-UDP/input/"
output_folder = "C:/Users/Admin/Documents/HK1 24-25/Du_an_cong_nghe/PDF2TEXT-UDP/output/"

# Chỉ định đường dẫn tới Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Hàm để chuyển đổi PDF scan thành văn bản
def read_pdf_scan(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# Hàm để đọc nội dung file văn bản
def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Hàm tính toán CER
def calculate_cer(ground_truth, predicted):
    return editdistance.eval(ground_truth, predicted) / len(ground_truth)

# Hàm tính toán WER
def calculate_wer(ground_truth_text, predicted_text):
    ground_truth_words = ground_truth_text.split()
    predicted_words = predicted_text.split()
    return editdistance.eval(ground_truth_words, predicted_words) / len(ground_truth_words)

# Đọc nội dung PDF scan
pdf_path = os.path.join(input_folder, "financial_statements.pdf")
ground_truth_text = read_pdf_scan(pdf_path)
print(ground_truth_text)

# Đọc các tệp txt từ thư mục output
predicted_files = [f for f in os.listdir(output_folder) if f.endswith('.txt')]

# Hàm tính toán các metrics
def evaluate_model_metrics(ground_truth, predicted, ground_truth_text, predicted_text):
    accuracy = accuracy_score(ground_truth, predicted)
    precision = precision_score(ground_truth, predicted, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predicted, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predicted, average='weighted', zero_division=0)
    cer = calculate_cer(ground_truth, predicted)
    wer = calculate_wer(ground_truth_text, predicted_text)
    return accuracy, precision, recall, f1, cer, wer

# Lưu trữ các kết quả metrics
all_metrics = {'File': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'CER': [], 'WER': []}

# Tính metrics cho mỗi tệp txt
for predicted_file in predicted_files:
    predicted_text = read_file_content(os.path.join(output_folder, predicted_file))
    
    # Chuyển văn bản thành các token
    ground_truth_tokens = list(ground_truth_text)
    predicted_tokens = list(predicted_text)
    
    # Cắt độ dài để khớp nếu cần
    min_len = min(len(ground_truth_tokens), len(predicted_tokens))
    ground_truth_tokens = ground_truth_tokens[:min_len]
    predicted_tokens = predicted_tokens[:min_len]
    
    # Tính các metrics
    accuracy, precision, recall, f1, cer, wer = evaluate_model_metrics(
        ground_truth_tokens, predicted_tokens, ground_truth_text, predicted_text)
    
    # Lưu kết quả cho từng file
    all_metrics['File'].append(predicted_file)
    all_metrics['Accuracy'].append(accuracy)
    all_metrics['Precision'].append(precision)
    all_metrics['Recall'].append(recall)
    all_metrics['F1 Score'].append(f1)
    all_metrics['CER'].append(cer)
    all_metrics['WER'].append(wer)
    
    # In kết quả
    print(f"Metrics for {predicted_file}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"CER: {cer}")
    print(f"WER: {wer}")
    print("-" * 40)

# Tạo DataFrame từ kết quả
metrics_df = pd.DataFrame(all_metrics)
print(metrics_df)

# Biểu đồ tổng hợp
average_metrics = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CER', 'WER']].mean()
plt.figure(figsize=(10, 5))
plt.bar(average_metrics.index, average_metrics.values)
plt.title("Average Model Performance Metrics")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.show()
