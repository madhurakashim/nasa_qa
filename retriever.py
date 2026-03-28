# =============================================================================
# File: retriever.py (Improved Retrieval)
# =============================================================================

import re
from src.embedder     import embed_single
from src.vector_store import search_similar, get_or_create_collection
from config           import TOP_K


# =============================================================================
# Configuration
# =============================================================================
DISTANCE_THRESHOLD = 0.65

#  Reduced effective top_k (even if config is higher)
EFFECTIVE_TOP_K = 3

CROSS_REF_PATTERN = re.compile(
    r'(?:Section|section|Chapter|chapter|see|See)\s+(\d+\.(?:\d+\.?)*)',
    re.IGNORECASE
)


# =============================================================================
# NEW: Reranking Function (Keyword Overlap)
# =============================================================================
def rerank_chunks(query: str, chunks: list) -> list:
    query_words = set(query.lower().split())

    def score(chunk):
        text_words = set(chunk["text"].lower().split())
        return len(query_words & text_words)

    return sorted(chunks, key=score, reverse=True)


# =============================================================================
# NEW: Filter Weak Chunks
# =============================================================================
def filter_chunks(chunks: list) -> list:
    return [
        c for c in chunks
        if len(c["text"]) > 150  # remove tiny/noisy chunks
    ]


# =============================================================================
# Cross Reference Detection (UNCHANGED)
# =============================================================================
def find_cross_references(chunks: list) -> list:

    referenced_sections = set()

    for chunk in chunks:
        matches = CROSS_REF_PATTERN.findall(chunk["text"])
        for match in matches:
            sec = match.rstrip(".")
            referenced_sections.add(sec)

    already_retrieved = set(
        c["section_number"] for c in chunks if c.get("section_number")
    )

    return list(referenced_sections - already_retrieved)


# =============================================================================
# Fetch Section Chunks (UNCHANGED)
# =============================================================================
def fetch_section_chunks(section_number: str) -> list:

    try:
        collection = get_or_create_collection()

        results = collection.get(
            where   = {"section_number": section_number},
            include = ["documents", "metadatas"]
        )

        if not results["documents"]:
            return []

        chunks = []

        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]

            chunks.append({
                "text":           doc,
                "page_number":    meta.get("page_number", 0),
                "section_number": meta.get("section_number", ""),
                "section_title":  meta.get("section_title", ""),
                "chunk_part":     meta.get("chunk_part", 1),
                "distance":       0.0,
                "source":         "cross_reference",
            })

        return chunks[:2]

    except Exception as e:
        print(f"Could not fetch section {section_number}: {e}")
        return []


# =============================================================================
# MAIN RETRIEVE FUNCTION (IMPROVED)
# =============================================================================
def retrieve(question: str, top_k: int = TOP_K) -> list:

    if not question.strip():
        return []

    print(f"Embedding and searching: '{question}'")

    # Step 1: Embed + Search (LIMIT top_k)
    question_vector = embed_single(question)
    raw_results     = search_similar(question_vector, top_k=EFFECTIVE_TOP_K)

    # Step 2: Distance filtering
    filtered = [c for c in raw_results if c["distance"] <= DISTANCE_THRESHOLD]

    if not filtered and raw_results:
        filtered = [raw_results[0]]

    # Step 3: Remove weak chunks
    filtered = filter_chunks(filtered)

    # Step 4: Rerank (VERY IMPORTANT)
    filtered = rerank_chunks(question, filtered)

    # Limit again after reranking
    filtered = filtered[:EFFECTIVE_TOP_K]

    # Step 5: Cross-reference detection
    cross_refs = find_cross_references(filtered)

    if cross_refs:
        print(f"Found cross-references to sections: {cross_refs}")

    # Step 6: Fetch referenced sections
    extra_chunks = []

    for sec_num in cross_refs[:2]:
        fetched = fetch_section_chunks(sec_num)
        extra_chunks.extend(fetched)

        if fetched:
            print(f"  Added {len(fetched)} chunks from Section {sec_num}")

    # Step 7: Combine
    all_chunks = filtered + extra_chunks

    # Step 8: Deduplicate
    seen_texts = set()
    deduped    = []

    for chunk in all_chunks:
        key = chunk["text"][:120]
        if key not in seen_texts:
            seen_texts.add(key)
            deduped.append(chunk)

    print(f"Retrieved {len(deduped)} chunks "
          f"({len(filtered)} direct + {len(extra_chunks)} cross-ref)")

    return deduped


# =============================================================================
# Wrapper (UNCHANGED)
# =============================================================================
def retrieve_with_context(question: str) -> dict:

    chunks = retrieve(question)

    return {
        "chunks":      chunks,
        "query_used":  question,
        "total_found": len(chunks),
    }