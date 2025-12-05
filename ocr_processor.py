import os
from paddleocr import PPStructureV3
from pathlib import Path

# Initialize the PP-StructureV3 pipeline for document parsing
# This pipeline handles tables, formulas, and restores reading order
pipeline = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

def process_pdf_to_markdown(pdf_path: Path) -> str:
    """
    Processes a PDF file using the PaddleOCR PPStructureV3 pipeline to extract
    structured content into a single Markdown string for LLM ingestion.
    """
    if not pdf_path.exists():
        return f"Error: PDF file not found at {pdf_path}"

    print(f"Starting parsing for: {pdf_path}")
    
    # PP-StructureV3's predict method returns results that can be saved as Markdown
    output = pipeline.predict(input=str(pdf_path))
    
    full_markdown_content = ""
    
    # Iterate through each page's output (assuming multiple pages are handled by the input)
    for i, res in enumerate(output):
        temp_md_path = f"temp_page_{i}.md"
        # Saves the structured Markdown, which includes table formatting
        res.save_to_markdown(save_path=temp_md_path) 
        
        with open(temp_md_path, 'r', encoding='utf-8') as f:
            markdown_chunk = f.read()
            full_markdown_content += f"\n\n### Requirements Document Page {i+1} ---\n"
            full_markdown_content += markdown_chunk
        
        os.remove(temp_md_path)
        
    return full_markdown_content

# --- Test Execution (Assuming your SRS PDF is named 'srs_document.pdf') ---
if __name__ == "__main__":
    # P2: Ensure your test PDF is in the data/ folder
    TEST_PDF_PATH = Path("data/srs_document.pdf") 

    markdown_output = process_pdf_to_markdown(TEST_PDF_PATH)
    
    if "Error" not in markdown_output:
        print("\n--- OCR SUCCESS: Extracted Requirements Markdown ---")
        print(markdown_output[:3000] + "...") # Print first 3000 chars for review
        print("\nSUCCESS: Ready for Phase 1.4 (ERNIE VLM Connector).")
    else:
        print(markdown_output)