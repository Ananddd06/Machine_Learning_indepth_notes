import re
import json
from PyPDF2 import PdfReader

pdf_path = "/Users/anand/Desktop/Ml_deep_notes_for_myself/scraping_Data/Andrew Ng ML Notes (1).pdf"
output_json = "/Users/anand/Desktop/Ml_deep_notes_for_myself/scraping_Data/book_structured.json"

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Remove control characters
            text = re.sub(r'[\x00-\x1F]+', '', text)
            # Normalize spaces
            text = re.sub(r'\s+', ' ', text)
            pages.append(text)
    return pages

# -----------------------------
# Detect multi-line Python code
# -----------------------------
def extract_python_code(text):
    code_blocks = []
    lines = text.splitlines()
    current_block = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")) or stripped.startswith("    "):
            current_block.append(line)
        elif "=" in stripped and not re.match(r"[A-Za-z]+\s*=", stripped):
            current_block.append(line)
        else:
            if current_block:
                code_blocks.append("\n".join(current_block))
                current_block = []
    if current_block:
        code_blocks.append("\n".join(current_block))
    return code_blocks

# -----------------------------
# Detect formulas
# -----------------------------
def extract_formulas(text):
    # Merge lines ending with operators to catch multi-line formulas
    text = re.sub(r'([^\s])\s*\n\s*', r'\1 ', text)
    pattern = r'([A-Za-z0-9φΣµθλπ∞±∑∏∫√≠≈≤≥⊂⊆∈∀∃]+\s*=\s*[^ \n]+)'
    return re.findall(pattern, text)

# -----------------------------
# Extract explanations
# -----------------------------
def extract_explanations(text, code_blocks, formulas):
    # Remove code and formulas
    cleaned = text
    for block in code_blocks:
        cleaned = cleaned.replace(block, "")
    for formula in formulas:
        cleaned = cleaned.replace(formula, "")
    # Split into paragraphs
    paragraphs = [p.strip() for p in cleaned.split('.') if len(p.strip()) > 20]
    return paragraphs

# -----------------------------
# Process a page
# -----------------------------
def process_text(text):
    code_blocks = extract_python_code(text)
    formulas = extract_formulas(text)
    explanations = extract_explanations(text, code_blocks, formulas)
    return {
        "content_preview": text[:500] + "...",
        "python_code": code_blocks,
        "formulas": formulas,
        "explanations": explanations
    }

# -----------------------------
# Main
# -----------------------------
pages = extract_pdf_text(pdf_path)
structured_data = {f"page_{i+1}": process_text(page) for i, page in enumerate(pages)}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4, ensure_ascii=False)

print(f"Structured JSON saved to {output_json}")
