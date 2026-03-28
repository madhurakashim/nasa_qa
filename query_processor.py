# =============================================================================
# File: query_processor.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Query Preprocessing & Normalization
#
# Description:
#   This module preprocesses and standardizes user queries before they are
#   embedded and passed to the retrieval system.
#
#   The quality of retrieval in a RAG pipeline is highly dependent on the
#   quality of the input query. This module improves query effectiveness by:
#     • Cleaning and normalizing text input
#     • Expanding domain-specific acronyms into full forms
#     • Ensuring consistent formatting for embedding models
#
# Why Query Processing Matters:
#   Raw user queries are often inconsistent, noisy, or incomplete:
#     Example: "  what is TRL??  "
#
#   Without preprocessing, this leads to:
#     • Poor embedding quality
#     • Lower semantic similarity scores
#     • Incorrect or irrelevant retrieval results
#
#   By transforming the query into:
#     "What is Technology Readiness Level?"
#   we significantly improve retrieval accuracy.
#
# Acronym Expansion:
#   Technical documents (like NASA handbooks) rely heavily on acronyms.
#   Embedding models perform better when given full semantic phrases rather
#   than abbreviations.
#
#   Example:
#     "TRL" → "Technology Readiness Level"
#
# Design Note:
#   Query preprocessing is often overlooked in basic RAG systems.
#   Including this module demonstrates attention to real-world robustness
#   and retrieval optimization.
#
# Key Features:
#     • Domain-specific acronym expansion
#     • Text normalization and formatting
#     • Robust handling of user input variations
#     • Modular pipeline for extensibility
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


# =============================================================================
# Domain-Specific Acronym Dictionary
# =============================================================================
# Maps commonly used NASA acronyms to their full semantic forms
NASA_ACRONYMS = {
    "TRL":  "Technology Readiness Level",
    "KDP":  "Key Decision Point",
    "SRR":  "System Requirements Review",
    "PDR":  "Preliminary Design Review",
    "CDR":  "Critical Design Review",
    "MDR":  "Mission Definition Review",
    "SDR":  "System Definition Review",
    "ORR":  "Operational Readiness Review",
    "FRR":  "Flight Readiness Review",
    "SAR":  "System Acceptance Review",
    "SEMP": "Systems Engineering Management Plan",
    "WBS":  "Work Breakdown Structure",
    "NASA": "National Aeronautics and Space Administration",
    "SE":   "Systems Engineering",
    "IV&V": "Independent Verification and Validation",
    "MOE":  "Measure of Effectiveness",
    "MOP":  "Measure of Performance",
    "ConOps": "Concept of Operations",
}


# =============================================================================
# Function: expand_acronyms
# =============================================================================
# Purpose:
#   Replaces known acronyms in the input text with their full forms.
#
# Description:
#   • Matches whole words only (avoids partial replacements)
#   • Preserves punctuation attached to words
#   • Improves semantic clarity for embedding models
#
# Parameters:
#   text (str) : Input query string
#
# Returns:
#   str : Text with acronyms expanded
#
# Example:
#   Input  : "What does TRL mean?"
#   Output : "What does Technology Readiness Level mean?"
# =============================================================================
def expand_acronyms(text: str) -> str:

    words  = text.split()
    result = []

    for word in words:
        stripped = word.strip("?.,!:;\"'()")

        if stripped.upper() in NASA_ACRONYMS:
            punctuation = word[len(stripped):]
            result.append(NASA_ACRONYMS[stripped.upper()] + punctuation)
        else:
            result.append(word)

    return " ".join(result)


# =============================================================================
# Function: clean_text
# =============================================================================
# Purpose:
#   Normalizes and formats raw user input.
#
# Description:
#   • Removes leading and trailing whitespace
#   • Collapses multiple spaces into a single space
#   • Capitalizes the first character
#   • Ensures the query ends with a question mark
#
# Parameters:
#   text (str) : Raw user input
#
# Returns:
#   str : Cleaned and standardized query string
#
# Example:
#   Input  : "  what is  TRL  "
#   Output : "What is TRL?"
# =============================================================================
def clean_text(text: str) -> str:

    text = text.strip()
    text = " ".join(text.split())

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Ensure question format
    if text and not text.endswith("?"):
        text = text + "?"

    return text


# =============================================================================
# Function: process_query
# =============================================================================
# Purpose:
#   Main entry point for query preprocessing.
#
# Description:
#   • Applies all transformation steps in a defined sequence
#   • Ensures query is clean, normalized, and semantically enriched
#   • Designed for direct use by the retrieval component
#
# Processing Pipeline:
#   1. Text cleaning (format normalization)
#   2. Acronym expansion (semantic enhancement)
#
# Parameters:
#   raw_question (str) : User's original query
#
# Returns:
#   str : Processed query ready for embedding
#
# Example:
#   Input  : "  what is TRL  "
#   Output : "What is Technology Readiness Level?"
# =============================================================================
def process_query(raw_question: str) -> str:

    if not raw_question or not raw_question.strip():
        return ""

    step1 = clean_text(raw_question)
    step2 = expand_acronyms(step1)
    

    return step2