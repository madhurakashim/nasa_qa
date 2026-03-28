# =============================================================================
# File: ingest.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Data Ingestion Pipeline
#
# Description:
#   This script runs the one-time ingestion pipeline that processes the
#   NASA handbook PDF and builds the vector database used for retrieval.
#
# Pipeline Overview:
#   PDF → Pages → Chunks → Embeddings → ChromaDB
#
# Why Separate Ingestion?
#   • Processing 270 pages takes ~1–2 minutes
#   • Running this at app startup would slow down UX
#   • Separation ensures fast, responsive UI during usage
#
# When to Run:
#   ✔ First-time setup
#   ✔ After updating PDF
#   ✔ After changing chunk size or embedding model
#   ✔ After deleting database
#
# Features:
#   • Step-by-step progress logging
#   • Execution time tracking
#   • Safe re-run protection (skip if already exists)
#   • Optional force re-ingestion flag
#
# Usage:
#   python ingest.py
#   python ingest.py --force
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


import time
import sys

from src.pdf_loader   import load_pdf, get_pdf_info
from src.chunker      import chunk_pages, get_chunk_stats
from src.embedder     import embed_texts
from src.vector_store import store_chunks, collection_exists, get_collection_stats
from config           import PDF_PATH


# =============================================================================
# Function: run_ingestion
# =============================================================================
# Purpose:
#   Executes the complete ingestion pipeline.
#
# Parameters:
#   force (bool):
#       • False → Skip ingestion if DB already exists
#       • True  → Rebuild database from scratch
#
# Steps:
#   1. Load PDF
#   2. Chunk text
#   3. Generate embeddings
#   4. Store in vector database
# =============================================================================
def run_ingestion(force: bool = False) -> None:

    print("=" * 55)
    print("  NASA Handbook — Ingestion Pipeline")
    print("=" * 55)

    # -------------------------------------------------------------------------
    # Guard: Skip ingestion if already exists
    # -------------------------------------------------------------------------
    if collection_exists() and not force:
        stats = get_collection_stats()

        print(f"\nVector store already exists!")
        print(f"  Chunks stored: {stats.get('total_chunks', '?')}")

        print(f"\nSkipping ingestion. To re-ingest, run:")
        print(f"  python ingest.py --force")

        print("\nYou can now start the app:")
        print("  streamlit run app.py")

        return


    total_start = time.time()


    # -------------------------------------------------------------------------
    # Step 1: Load PDF
    # -------------------------------------------------------------------------
    print("\nStep 1/4 — Loading PDF...")

    info = get_pdf_info(PDF_PATH)

    print(f"  File:  {PDF_PATH}")
    print(f"  Size:  {info.get('file_size_mb', '?')} MB")
    print(f"  Pages: {info.get('total_pages', '?')}")

    t0 = time.time()
    pages = load_pdf(PDF_PATH)

    print(f"  Loaded {len(pages)} pages")
    print(f"  Done in {round(time.time() - t0, 1)}s")


    # -------------------------------------------------------------------------
    # Step 2: Chunk Text
    # -------------------------------------------------------------------------
    print("\nStep 2/4 — Chunking text...")

    t0 = time.time()
    chunks = chunk_pages(pages)

    stats = get_chunk_stats(chunks)

    print(f"  Total chunks created : {stats['total_chunks']}")
    print(f"  Avg chunk size       : {stats['avg_length_chars']} chars")
    print(f"  Pages covered        : {stats['unique_pages']}")
    print(f"  Done in {round(time.time() - t0, 1)}s")


    # -------------------------------------------------------------------------
    # Step 3: Generate Embeddings
    # -------------------------------------------------------------------------
    print("\nStep 3/4 — Generating embeddings...")
    print("  (First run downloads model ~80MB, one-time only)")

    t0 = time.time()

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print(f"  Embedded {len(embeddings)} chunks")

    if embeddings:
        print(f"  Vector dimension: {len(embeddings[0])}")

    print(f"  Done in {round(time.time() - t0, 1)}s")


    # -------------------------------------------------------------------------
    # Step 4: Store in ChromaDB
    # -------------------------------------------------------------------------
    print("\nStep 4/4 — Storing in ChromaDB...")

    t0 = time.time()
    store_chunks(chunks, embeddings)

    print(f"  Done in {round(time.time() - t0, 1)}s")


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = round(time.time() - total_start, 1)

    print("\n" + "=" * 55)
    print(f"  Ingestion complete in {total_time}s")
    print(f"  {len(chunks)} chunks ready for retrieval")
    print("=" * 55)

    print("\nNext step:")
    print("  streamlit run app.py\n")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":

    force = "--force" in sys.argv

    if force:
        print("Force mode enabled — rebuilding vector database")

    run_ingestion(force=force)