# =============================================================================
# File: config.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
#
# Description:
#   Central configuration file for the entire QA system.
#   All tunable parameters are defined here for easy modification
#   without changing core logic.
#
# Why this file matters:
#   • Keeps system modular and maintainable
#   • Enables quick experimentation (chunk size, models, etc.)
#   • Avoids hardcoding values across multiple files
#
# =============================================================================


# =============================================================================
# Data Paths
# =============================================================================
# Path to source PDF document
PDF_PATH = "data/nasa_handbook.pdf"

# Path where ChromaDB will persist embeddings
DB_PATH = "db/chroma_store"

# Name of the vector collection inside ChromaDB
COLLECTION_NAME = "nasa_handbook"


# =============================================================================
# Embedding Model
# =============================================================================
# Model: all-MiniLM-L6-v2
# • Size: ~80MB (lightweight)
# • Fast and efficient for semantic search
# • Good balance between speed and accuracy
# • Works fully offline (no API required)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# LLM Configuration (Ollama)
# =============================================================================
# Model: llama3.2 (local)
# • Runs via Ollama (no API key required)
# • Good reasoning capability for QA tasks
# • Fully offline → secure + free
#
# Setup command:
#   ollama pull llama3.2
LLM_MODEL = "llama3.2"


# =============================================================================
# Chunking Strategy
# =============================================================================
# Why chunking matters:
#   • LLMs have context limits
#   • Large documents must be split into smaller parts
#
# CHUNK_SIZE = 1000
#   • Increased from 500 → preserves full section context
#   • Prevents splitting important concepts (e.g., Vee Model)
#
# CHUNK_OVERLAP = 200
#   • Ensures continuity between chunks
#   • Prevents loss of boundary information
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


# =============================================================================
# Retrieval Configuration
# =============================================================================
# Number of chunks retrieved per query
#
# TOP_K = 6
#   • Increased from 4 → improves recall
#   • Allows inclusion of cross-referenced sections
#   • Balanced to avoid too much noise
TOP_K = 6