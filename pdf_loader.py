# =============================================================================
# File: pdf_loader.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : PDF Ingestion & Text Extraction
#
# Description:
#   This module is responsible for loading the NASA Systems Engineering
#   Handbook PDF and extracting text content on a per-page basis.
#
#   The extracted data is structured to preserve page-level granularity,
#   which is critical for:
#     • Accurate citation in generated answers
#     • Traceability of retrieved information
#     • Improved retrieval precision in RAG pipelines
#
# Why Page-Level Extraction:
#   Extracting the entire document as a single text block would lose
#   positional context. By processing page-by-page, we retain:
#     • Exact page references for citations
#     • Better alignment between retrieval results and source content
#
# Library Used:
#   PyMuPDF (imported as 'fitz')
#     • High-performance PDF parsing
#     • Efficient text extraction per page
#     • Supports metadata retrieval
#
# Key Features:
#     • Page-wise text extraction with indexing
#     • Skips blank/non-text pages automatically
#     • File existence validation with clear error messaging
#     • Lightweight metadata extraction for debugging/UI display
#
# Dependencies:
#     • pymupdf (fitz)
#     • os (file handling)
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


import fitz  # PyMuPDF — installed as 'pymupdf' but imported as 'fitz'
import os


# =============================================================================
# Function: load_pdf
# =============================================================================
# Purpose:
#   Loads a PDF document and extracts text content page-by-page.
#
# Description:
#   • Validates file existence before processing
#   • Iterates through each page of the PDF
#   • Extracts plain text using PyMuPDF
#   • Skips pages with no textual content
#   • Associates each text block with its page number
#
# Parameters:
#   pdf_path (str) : Absolute or relative path to the PDF file
#
# Returns:
#   list[dict] :
#       A list of page objects, each containing:
#           {
#               "page_number": int,
#               "text": str
#           }
#
# Raises:
#   FileNotFoundError : If the specified PDF path does not exist
#
# Notes:
#   • Page numbering is 1-based (human-readable)
#   • Blank pages (e.g., images or separators) are excluded
# =============================================================================
def load_pdf(pdf_path: str) -> list[dict]:

    # Validate file existence
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}\n"
            f"Please download the NASA handbook and place it at {pdf_path}"
        )

    print(f"Opening PDF: {pdf_path}")

    # Open PDF document
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"Total pages in PDF: {total_pages}")

    pages = []

    # Iterate through all pages
    for page_index in range(total_pages):
        page = doc[page_index]
        text = page.get_text()

        # Skip pages without meaningful text
        if text.strip():
            pages.append({
                "page_number": page_index + 1,  # Convert to 1-based indexing
                "text": text
            })

    # Close document to free resources
    doc.close()

    print(f"Pages with text extracted: {len(pages)} / {total_pages}")

    return pages


# =============================================================================
# Function: get_pdf_info
# =============================================================================
# Purpose:
#   Retrieves basic metadata and diagnostic information about the PDF.
#
# Description:
#   • Extracts total page count
#   • Computes file size in megabytes
#   • Retrieves document title (if available)
#   • Useful for debugging, validation, and UI display
#
# Parameters:
#   pdf_path (str) : Path to the PDF file
#
# Returns:
#   dict :
#       {
#           "total_pages": int,
#           "file_size_mb": float,
#           "title": str
#       }
#
# Notes:
#   • Returns an error dictionary if file is not found
# =============================================================================
def get_pdf_info(pdf_path: str) -> dict:

    if not os.path.exists(pdf_path):
        return {"error": "PDF not found"}

    doc = fitz.open(pdf_path)

    info = {
        "total_pages": len(doc),
        "file_size_mb": round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
        "title": doc.metadata.get("title", "Unknown"),
    }

    doc.close()

    return info