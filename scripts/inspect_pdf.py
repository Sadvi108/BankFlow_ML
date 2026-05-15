
import pdfplumber
import sys
import os

# Fix encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def inspect_pdf(filename):
    print(f"Inspecting {filename}...")
    try:
        with pdfplumber.open(filename) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                print(f"--- Page {i+1} ---")
                if text:
                    print(text)
                else:
                    print("[No text extracted]")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_pdf("Receipts/PUBLIC BANK - INVOICE.pdf")
