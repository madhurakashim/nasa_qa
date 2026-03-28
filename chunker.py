# =============================================================================
# File: chunker.py
# =============================================================================
# Intelligent Section-Aware Chunking (Improved Version)
# =============================================================================

import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


# =============================================================================
# Improved Regex Patterns (More Robust)
# =============================================================================
SECTION_HEADING_PATTERN = re.compile(
    r'^\s*(\d+(?:\.\d+)*)\s+(.{3,80})',
    re.MULTILINE
)

APPENDIX_PATTERN = re.compile(
    r'^\s*(Appendix\s+[A-Z][\w\.]*)\s*[:\-]*\s*(.{0,80})',
    re.MULTILINE
)


# =============================================================================
# Utility: Detect Table Blocks
# =============================================================================
'''def looks_like_table(text: str) -> bool:
    return (
        "Term" in text and "Definition" in text
    )'''
def looks_like_table(text: str) -> bool:
    return (
        "Term" in text and "Definition" in text
    ) or (
        text.count("\n") > 5 and "|" in text
    )


# =============================================================================
# Utility: Safe Sentence Boundary Split
# =============================================================================
'''def safe_split(text: str, start: int, chunk_size: int) -> tuple:
    end = start + chunk_size
    chunk = text[start:end]

    # Try to end at sentence boundary
    last_period = chunk.rfind(".")
    if last_period > 200:
        end = start + last_period + 1
        chunk = text[start:end]

    return chunk, end'''
def safe_split(text: str, start: int, chunk_size: int) -> tuple:
    end = min(start + chunk_size, len(text))
    chunk = text[start:end]

    # Try multiple sentence boundaries
    boundaries = [".", "\n", ";", ":"]

    best_pos = -1
    for b in boundaries:
        pos = chunk.rfind(b)
        if pos > best_pos:
            best_pos = pos

    if best_pos > 100:  # avoid too small chunks
        end = start + best_pos + 1
        chunk = text[start:end]

    return chunk, end


# =============================================================================
# Extract Section Number
# =============================================================================
def extract_section_number(text: str) -> str:
    match = SECTION_HEADING_PATTERN.search(text)
    if match:
        return match.group(1)

    match = APPENDIX_PATTERN.search(text)
    if match:
        return match.group(1)

    return ""


# =============================================================================
# Extract Section Title
# =============================================================================
def extract_section_title(text: str) -> str:
    match = SECTION_HEADING_PATTERN.search(text)
    if match:
        return match.group(2).strip()

    match = APPENDIX_PATTERN.search(text)
    if match:
        return match.group(2).strip()

    return ""


# =============================================================================
# Split Page into Sections
# =============================================================================
def split_into_sections(full_text: str, page_number: int) -> list:

    h_matches = list(SECTION_HEADING_PATTERN.finditer(full_text))
    a_matches = list(APPENDIX_PATTERN.finditer(full_text))
    all_matches = sorted(h_matches + a_matches, key=lambda m: m.start())

    if not all_matches:
        return [{
            "text": full_text.strip(),
            "page_number": page_number,
            "section_number": "",
            "section_title": "",
        }]

    sections = []

    # Pre-heading text
    if all_matches[0].start() > 0:
        pre = full_text[:all_matches[0].start()].strip()
        if pre:
            sections.append({
                "text": pre,
                "page_number": page_number,
                "section_number": "",
                "section_title": "",
            })

    # Extract sections
    for i, match in enumerate(all_matches):
        start = match.start()
        end = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(full_text)

        sec_text = full_text[start:end].strip()
        if not sec_text:
            continue

        try:
            sec_num = match.group(1)
            sec_title = match.group(2).strip()
        except IndexError:
            sec_num, sec_title = "", ""

        sections.append({
            "text": sec_text,
            "page_number": page_number,
            "section_number": sec_num,
            "section_title": sec_title,
        })

    return sections


# =============================================================================
# Chunk Sections Safely
# =============================================================================
def chunk_section(section: dict, chunk_size: int, overlap: int) -> list:

    text = section["text"]

    # ✅ FIX: Preserve tables
    if looks_like_table(text):
        section["chunk_part"] = 1
        return [section]
    # Trim very large sections
    if len(text) > 3 * chunk_size:
        text = text[:3 * chunk_size]
    # Small section → no split
    if len(text) <= chunk_size:
        section["chunk_part"] = 1
        return [section]
    
    sub_chunks = []
    start = 0
    part_num = 1

    while start < len(text):
        chunk_text, new_end = safe_split(text, start, chunk_size)

        if chunk_text.strip():
            sub_chunks.append({
                "text": chunk_text,
                "page_number": section["page_number"],
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "chunk_part": part_num,
            })
            part_num += 1

        # Move with overlap
        start = max(new_end - overlap, start + 1)

        # Prevent infinite loop
        if start <= 0:
            break

    return sub_chunks


# =============================================================================
# Main Pipeline
# =============================================================================
def chunk_pages(pages: list) -> list:

    all_chunks = []

    for page in pages:
        sections = split_into_sections(page["text"], page["page_number"])

        for section in sections:
            sub_chunks = chunk_section(section, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in sub_chunks:
                sec_num = chunk.get("section_number", "")
                part_num = chunk.get("chunk_part", 1)

                sec_part = f"_sec{sec_num.replace('.', '_')}" if sec_num else ""
                chunk_id = f"page{page['page_number']}{sec_part}_part{part_num}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "page_number": page["page_number"],
                    "section_number": sec_num,
                    "section_title": chunk.get("section_title", ""),
                    "chunk_part": part_num,
                })

    return all_chunks


# =============================================================================
# Stats
# =============================================================================
def get_chunk_stats(chunks: list) -> dict:

    if not chunks:
        return {"total_chunks": 0}

    lengths = [len(c["text"]) for c in chunks]
    sectioned = [c for c in chunks if c.get("section_number")]

    return {
        "total_chunks": len(chunks),
        "chunks_with_sections": len(sectioned),
        "avg_length_chars": round(sum(lengths) / len(lengths)),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "unique_pages": len(set(c["page_number"] for c in chunks)),
        "unique_sections": len(set(
            c["section_number"] for c in chunks if c.get("section_number")
        )),
    }