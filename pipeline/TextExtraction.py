import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"add your tesseract path here"  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# PDF path
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of this script
pdf_path = os.path.join(script_dir, "Test_Set_NOimg.pdf")  # ensures correct location

# Poppler bin path
poppler_path = r"add your poppler path here"  # e.g., r"C:\path\to\poppler\bin"

# Convert PDF to images (important: add poppler_path)
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

# Store all text
all_text = ""

# Loop through each page
for i, image in enumerate(pages):
    print(f"\n--- Page {i + 1} ---")

    # OCR for plain text
    raw_text = pytesseract.image_to_string(image)
    all_text += f"\n\n--- Page {i + 1} ---\n{raw_text}"

    print(f"\n📝 Extracted Text:\n{raw_text[:500]}")

    # OCR with layout info
    layout_data = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Table detection (basic)
    lines = raw_text.split("\n")
    table_like = [line for line in lines if "\t" in line or line.count(" ") > 5]
    print("\n📊 Table-like Content:")
    for row in table_like:
        print(row)

    # Layout analysis
    print("\n📐 Layout Elements (text boxes):")
    for j, text in enumerate(layout_data["text"]):
        if int(layout_data["conf"][j]) > 60 and text.strip():
            print(f"→ Text: '{text}' at (x={layout_data['left'][j]}, y={layout_data['top'][j]})")

# Save full OCR result
with open("full_pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("\n✅ All page text saved to 'full_pdf_text.txt'")
