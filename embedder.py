# =============================================================================
# File: embedder.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Embedding Generation (Semantic Encoding)
#
# Description:
#   This module converts textual data into dense vector representations
#   (embeddings) using a pre-trained Sentence Transformer model.
#
#   These embeddings capture semantic meaning, enabling similarity-based
#   retrieval in downstream components such as vector databases (e.g., ChromaDB).
#
#   Instead of relying on exact keyword matching, embeddings allow the system
#   to identify conceptually similar content, significantly improving retrieval
#   accuracy in question-answering pipelines.
#
# What is an Embedding:
#   • A numerical vector representation of text (e.g., [0.21, -0.54, ...])
#   • Encodes semantic meaning in high-dimensional space
#   • Similar texts → vectors are closer (low distance)
#   • Dissimilar texts → vectors are farther apart
#
# Model Used:
#   all-MiniLM-L6-v2 (via SentenceTransformers)
#     • Lightweight (~80MB)
#     • Fast inference on CPU
#     • Optimized for semantic similarity tasks
#     • Works offline after initial download
#
# Optimization – Model Caching:
#   Loading transformer models is computationally expensive.
#   This module uses a module-level cache to:
#     • Load the model only once per session
#     • Reuse the same instance across multiple calls
#     • Improve overall system performance
#
# Key Features:
#     • Batch embedding for efficiency
#     • Single-query embedding for real-time retrieval
#     • Automatic conversion to Python-native formats
#     • Progress tracking for large workloads
#
# Dependencies:
#     • sentence-transformers
#     • config.py (EMBEDDING_MODEL)
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


# =============================================================================
# Module-Level Model Cache
# =============================================================================
# Holds the loaded embedding model instance to avoid repeated initialization
_model = None


# =============================================================================
# Function: get_model
# =============================================================================
# Purpose:
#   Loads and returns the embedding model instance.
#
# Description:
#   • Initializes the SentenceTransformer model on first invocation
#   • Stores the model in a module-level cache
#   • Returns cached instance for subsequent calls
#
# Performance Benefit:
#   Avoids repeated model loading (~5 seconds overhead per load)
#
# Returns:
#   SentenceTransformer : Loaded embedding model
# =============================================================================
def get_model() -> SentenceTransformer:

    global _model

    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        print("(This takes ~5 seconds on first load, then it's cached)")

        _model = SentenceTransformer(EMBEDDING_MODEL)

        print("Embedding model loaded.")

    return _model


# =============================================================================
# Function: embed_texts
# =============================================================================
# Purpose:
#   Converts a list of text inputs into embedding vectors.
#
# Description:
#   • Uses batch processing for efficient encoding
#   • Generates dense vector representations for each input string
#   • Converts output from NumPy arrays to Python lists for compatibility
#
# Parameters:
#   texts (list[str]) : List of input text strings
#
# Returns:
#   list[list[float]] :
#       A list of embedding vectors (one per input text)
#
# Example:
#   Input  : ["What is the Vee Model?", "Systems engineering is..."]
#   Output : [[0.21, -0.54, ...], [0.33, 0.12, ...]]
#
# Notes:
#   • Each vector has fixed dimensionality (e.g., 384 for MiniLM)
#   • Individual values are not interpretable — similarity is key
# =============================================================================
def embed_texts(texts: list[str]) -> list[list[float]]:

    model = get_model()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,   # Displays progress for large batches
        batch_size=32,             # Balanced for CPU performance & memory usage
        normalize_embeddings=True
    )

    # Convert NumPy array to Python list (required by vector DBs like Chroma)
    return embeddings.tolist()


# =============================================================================
# Function: embed_single
# =============================================================================
# Purpose:
#   Generates an embedding for a single text input.
#
# Description:
#   • Wrapper around embed_texts() for convenience
#   • Primarily used for encoding user queries during retrieval
#
# Parameters:
#   text (str) : Input query string
#
# Returns:
#   list[float] : Embedding vector corresponding to the input text
#
# Usage:
#   query_vector = embed_single("What is the Vee Model?")
# =============================================================================
def embed_single(text: str) -> list[float]:

    return embed_texts([text])[0]