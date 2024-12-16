import pandas as pd
from difflib import SequenceMatcher

def calculate_similarity(text1, text2, threshold=0.8):
    """
    Calculate the similarity between two strings, text1 and text2.
    Returns True if the similarity is >= threshold, otherwise False.
    """
    similarity = SequenceMatcher(None, str(text1), str(text2)).ratio()
    return similarity >= threshold

def evaluate_table_accuracy(model_csv, ground_truth_csv, threshold=0.8):
    """
    Evaluate the accuracy of a detected table compared to the ground truth table.
    """
    # Read the two tables from CSV files
    model_df = pd.read_csv(model_csv, header=None)  # No guarantee of headers
    ground_truth_df = pd.read_csv(ground_truth_csv, header=None)

    # Convert the entire table contents to a flattened list of text
    model_cells = model_df.fillna("").values.flatten()
    ground_truth_cells = ground_truth_df.fillna("").values.flatten()

    # Compare each cell between the two tables
    total_cells = len(ground_truth_cells)  # Total number of cells in the ground truth
    match_count = 0  # Number of cells that match or are similar

    for gt_cell in ground_truth_cells:
        for model_cell in model_cells:
            if calculate_similarity(model_cell, gt_cell, threshold):
                match_count += 1
                break  # Stop comparing the current cell once a match is found

    # Calculate accuracy
    accuracy = match_count / total_cells if total_cells > 0 else 0
    return accuracy

# Example usage
model_csv = "./PDF2TEXT-UDP/output/page_2_table_1.csv"
ground_truth_csv = "./PDF2TEXT-UDP/ground_truth/gt_page_2_table_1.csv"
accuracy = evaluate_table_accuracy(model_csv, ground_truth_csv, threshold=0.8)
print(f"Accuracy: {accuracy * 100:.2f}%")
