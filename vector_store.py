# =============================================================================
# File: vector_store.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Vector Storage & Similarity Search (ChromaDB)
#
# Description:
#   This module manages storage and retrieval of document chunks using
#   ChromaDB, a vector database optimized for semantic search.
#
#   It enables:
#     • Persistent storage of embeddings
#     • Fast similarity search using vector indexing (HNSW)
#     • Metadata-based retrieval for precise citations
#
# Core Responsibilities:
#     • Initialize ChromaDB persistent client
#     • Create or load vector collection
#     • Store chunk embeddings with metadata
#     • Perform similarity search for query embeddings
#     • Provide collection status and statistics
#
# Why ChromaDB?
#   • Lightweight and easy to integrate
#   • Supports persistent storage (disk-based)
#   • Built-in vector indexing (HNSW)
#   • Efficient similarity search at scale
#
# Similarity Metric:
#   Cosine Similarity is used:
#     • Measures angle between vectors, not magnitude
#     • Works well for text embeddings
#     • Robust to sentence length variations
#
# Metadata Stored (Critical for QA System):
#     • page_number
#     • section_number
#     • section_title
#     • chunk_part
#
#   This enables precise answers like:
#     "Section 2.1 (Page 13) explains..."
#
# Dependencies:
#     • chromadb
#     • config.py (DB_PATH, COLLECTION_NAME)
#
# Author  : <Your Name>
# Version : 1.0
# =============================================================================


import chromadb
from config import DB_PATH, COLLECTION_NAME


# =============================================================================
# Function: get_client
# =============================================================================
# Purpose:
#   Initializes and returns a persistent ChromaDB client.
#
# Description:
#   • Uses disk-based storage at DB_PATH
#   • Ensures embeddings are retained across sessions
#
# Returns:
#   chromadb.Client
# =============================================================================
def get_client():
    return chromadb.PersistentClient(path=DB_PATH)


# =============================================================================
# Function: get_or_create_collection
# =============================================================================
# Purpose:
#   Retrieves existing collection or creates a new one if absent.
#
# Description:
#   • Uses HNSW index for efficient nearest neighbor search
#   • Configured with cosine similarity metric
#
# Why Cosine Similarity?
#   • Captures semantic similarity effectively
#   • Independent of vector magnitude
#   • Ideal for text embeddings
#
# Returns:
#   chromadb.Collection
# =============================================================================
def get_or_create_collection():

    client = get_client()

    return client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )


# =============================================================================
# Function: store_chunks
# =============================================================================
# Purpose:
#   Stores document chunks along with their embeddings in ChromaDB.
#
# Description:
#   • Accepts processed chunks and corresponding embeddings
#   • Stores text, vector, and metadata together
#   • Metadata enables citation-aware retrieval
#
# Parameters:
#   chunks     (list[dict]) : Preprocessed document chunks
#   embeddings (list[list]) : Corresponding embedding vectors
#
# Metadata Stored:
#   • page_number
#   • section_number
#   • section_title
#   • chunk_part
#
# Returns:
#   None
# =============================================================================
def store_chunks(chunks: list, embeddings: list) -> None:

    collection = get_or_create_collection()

    metadatas = []

    for c in chunks:
        metadatas.append({
            "page_number":    c["page_number"],
            "section_number": c.get("section_number", ""),
            "section_title":  c.get("section_title", ""),
            "chunk_part":     c.get("chunk_part", 1),
        })

    collection.add(
        ids        = [c["chunk_id"] for c in chunks],
        embeddings = embeddings,
        documents  = [c["text"]     for c in chunks],
        metadatas  = metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at: {DB_PATH}")


# =============================================================================
# Function: search_similar
# =============================================================================
# Purpose:
#   Retrieves top-k most semantically similar chunks for a query.
#
# Description:
#   • Performs vector similarity search using query embedding
#   • Returns ranked results with metadata and distance score
#
# Distance Interpretation:
#   • < 0.30 → Highly relevant
#   • 0.30–0.60 → Moderately relevant
#   • > 0.60 → Likely irrelevant
#
# Parameters:
#   query_embedding (list) : Query vector
#   top_k           (int)  : Number of results to return
#
# Returns:
#   list[dict] :
#       [
#         {
#           "text": str,
#           "page_number": int,
#           "section_number": str,
#           "section_title": str,
#           "chunk_part": int,
#           "distance": float
#         }
#       ]
# =============================================================================
def search_similar(query_embedding: list, top_k: int) -> list:

    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []

    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]

        chunks.append({
            "text":           results["documents"][0][i],
            "page_number":    meta.get("page_number", 0),
            "section_number": meta.get("section_number", ""),
            "section_title":  meta.get("section_title", ""),
            "chunk_part":     meta.get("chunk_part", 1),
            "distance":       round(results["distances"][0][i], 4),
        })

    return chunks


# =============================================================================
# Function: collection_exists
# =============================================================================
# Purpose:
#   Checks whether the vector collection has data.
#
# Description:
#   • Useful before running queries
#   • Prevents errors when DB is empty
#
# Returns:
#   bool : True if collection exists and has data
# =============================================================================
def collection_exists() -> bool:

    try:
        client     = get_client()
        collection = client.get_collection(name=COLLECTION_NAME)

        return collection.count() > 0

    except Exception:
        return False


# =============================================================================
# Function: get_collection_stats
# =============================================================================
# Purpose:
#   Provides diagnostic information about the vector database.
#
# Description:
#   • Returns total stored chunks
#   • Includes collection name and DB path
#   • Helps debugging and monitoring
#
# Returns:
#   dict :
#       {
#           "total_chunks": int,
#           "collection": str,
#           "db_path": str
#       }
# =============================================================================
def get_collection_stats() -> dict:

    try:
        client     = get_client()
        collection = client.get_collection(name=COLLECTION_NAME)

        return {
            "total_chunks": collection.count(),
            "collection":   COLLECTION_NAME,
            "db_path":      DB_PATH,
        }

    except Exception:
        return {
            "error": "Collection not found. Run ingest.py first."
        }