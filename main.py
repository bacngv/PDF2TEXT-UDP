from pdf_processing import process_pdf

if __name__ == "__main__":
    pdf_path = "./PDF2TEXT-UDP/input/financial_statements.pdf"  # Path to your PDF
    output_folder = "./PDF2TEXT-UDP/output"  # Output folder
    process_pdf(pdf_path, output_folder)
