# =============================================================================
# File: answer_generator.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Answer Generation (LLM Interface)
#
# Description:
#   This module is responsible for generating precise, citation-backed answers
#   using a locally hosted Large Language Model (via Ollama).
#
#   It performs the following key functions:
#     • Constructs a structured prompt from retrieved document chunks
#     • Differentiates primary and cross-referenced content
#     • Enforces strict grounding in provided context (no hallucinations)
#     • Invokes the LLM to generate concise, fact-based responses
#     • Extracts and returns citation metadata (page & section numbers)
#
# Key Features:
#     • Section-aware citation formatting
#     • Cross-reference handling for multi-hop reasoning
#     • Deterministic response generation (low temperature)
#     • Robust error handling for LLM invocation
#
# Dependencies:
#     • ollama (local LLM runtime)
#     • config.py (model configuration)
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


import ollama
from config import LLM_MODEL


# =============================================================================
# Function: build_prompt
# =============================================================================
# Purpose:
#   Constructs a structured and context-rich prompt for the LLM using
#   retrieved document chunks.
#
# Description:
#   • Formats each chunk with page number, section number, and title
#   • Clearly distinguishes cross-referenced sections from primary content
#   • Aggregates all chunks into a unified context block
#   • Embeds strict instructions to ensure grounded, citation-based answers
#
# Parameters:
#   question (str) : User query
#   chunks   (list): Retrieved document chunks with metadata
#
# Returns:
#   str : Fully formatted prompt ready for LLM consumption
# =============================================================================
def build_prompt(question: str, chunks: list) -> str:

    context_parts = []

    for chunk in chunks:
        page_num   = chunk.get("page_number", "?")
        sec_num    = chunk.get("section_number", "")
        sec_title  = chunk.get("section_title", "")
        is_xref    = chunk.get("source") == "cross_reference"

        # Construct source label
        if sec_num and sec_title:
            source_label = f"Page {page_num}, Section {sec_num} ({sec_title})"
        elif sec_num:
            source_label = f"Page {page_num}, Section {sec_num}"
        else:
            source_label = f"Page {page_num}"

        if is_xref:
            source_label += " [cross-referenced section]"

        context_parts.append(f"[{source_label}]\n{chunk['text'].strip()}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert assistant for the NASA Systems Engineering Handbook.

Use ONLY the context below to answer the question.
Always cite the exact page number AND section number in your answer.
If context comes from multiple sections, mention all of them.
If the answer is not in the context, say:
"I could not find this information in the provided sections."
Do not invent information. Be concise and precise.

=== CONTEXT FROM THE HANDBOOK ===
{context}
=== END OF CONTEXT ===

Question: {question}

Answer (cite page and section numbers):"""

    return prompt


# =============================================================================
# Function: generate_answer
# =============================================================================
# Purpose:
#   Generates a final answer using the LLM based on retrieved context.
#
# Description:
#   • Validates input chunks before processing
#   • Calls the local Ollama model with a structured prompt
#   • Controls response quality using deterministic parameters
#   • Handles runtime and API-level exceptions gracefully
#   • Extracts and returns citation metadata for UI display
#
# Parameters:
#   question (str) : User query
#   chunks   (list): Retrieved document chunks
#
# Returns:
#   dict :
#       {
#           "answer"          : Generated response text
#           "source_pages"    : List of cited page numbers
#           "source_sections" : List of cited section numbers
#           "chunks_used"     : Raw chunks (for traceability/UI)
#           "model_used"      : LLM model identifier
#       }
# =============================================================================
def generate_answer(question: str, chunks: list) -> dict:

    if not chunks:
        return {
            "answer":          "No relevant sections found in the handbook for your question.",
            "source_pages":    [],
            "source_sections": [],
            "chunks_used":     [],
            "model_used":      LLM_MODEL,
        }

    prompt = build_prompt(question, chunks)

    print(f"Sending to Ollama ({LLM_MODEL})...")
    print("(First call may take 15-20 seconds while model loads into memory)")

    try:
        response = ollama.chat(
            model    = LLM_MODEL,
            messages = [{"role": "user", "content": prompt}],
            options  = {
                "temperature": 0.1,
                "num_predict": 600,
            }
        )
        answer_text = response["message"]["content"]

    except ollama.ResponseError as e:
        answer_text = (
            f"Ollama error: {str(e)}\n\n"
            f"Make sure:\n"
            f"1. Ollama app is running\n"
            f"2. You ran: ollama pull {LLM_MODEL}"
        )

    except Exception as e:
        answer_text = f"Unexpected error: {str(e)}"

    # Extract citation metadata
    source_pages = sorted(set(
        c["page_number"] for c in chunks if c.get("page_number")
    ))

    source_sections = sorted(set(
        c["section_number"] for c in chunks if c.get("section_number")
    ))

    return {
        "answer":          answer_text,
        "source_pages":    source_pages,
        "source_sections": source_sections,
        "chunks_used":     chunks,
        "model_used":      LLM_MODEL,
    }