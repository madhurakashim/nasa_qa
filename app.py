# =============================================================================
# File: app.py
# =============================================================================
# Project : NASA Systems Engineering Handbook QA System
# Module  : Streamlit UI (User Interface Layer)
#
# Description:
#   This is the frontend of the QA system built using Streamlit.
#   It allows users to interact with the system in natural language
#   and view answers along with precise citations and retrieved context.
#
# Core Responsibilities:
#     • Accept user queries
#     • Display system status and configuration
#     • Trigger retrieval + answer generation pipeline
#     • Show answers with page & section citations
#     • Visualize retrieved chunks (direct + cross-referenced)
#
# Key Features:
#     • Clean UI with sidebar status panel
#     • Query preprocessing (acronym expansion)
#     • Citation-aware answers (page + section)
#     • Cross-reference visualization (advanced feature)
#     • Expandable context for transparency
#
# Tech Stack:
#     • Streamlit (UI)
#     • ChromaDB (Vector Store)
#     • Sentence Transformers (Embeddings)
#     • Ollama (Local LLM)
#
# Author  : <Madhura D>
# Version : 1.0
# =============================================================================


import streamlit as st
from src.query_processor  import process_query
from src.retriever        import retrieve
from src.answer_generator import generate_answer
from src.vector_store     import collection_exists, get_collection_stats
from config               import PDF_PATH, LLM_MODEL, EMBEDDING_MODEL, TOP_K


# =============================================================================
# Page Setup
# =============================================================================
st.set_page_config(
    page_title="NASA Handbook QA",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NASA Systems Engineering Handbook — QA")
st.markdown("Ask questions in plain English. Answers include **page and section citations**.")
st.divider()


# =============================================================================
# Sidebar: System Status & Config
# =============================================================================
with st.sidebar:

    st.header("⚙️ System Status")

    if collection_exists():
        stats = get_collection_stats()
        st.success("Vector store ready")
        st.metric("Chunks indexed", stats.get("total_chunks", "?"))
    else:
        st.error("Vector store not found")
        st.warning("Run:\n\n`python ingest.py`")

    st.divider()

    st.header("🔧 Configuration")
    st.code(
        f"Model:      {LLM_MODEL}\n"
        f"Embeddings: {EMBEDDING_MODEL}\n"
        f"Top-K:      {TOP_K}"
    )

    st.divider()

    st.header("💡 Sample Questions")

    samples = [
        "What is the Vee Model?",
        "What are the entry criteria for PDR?",
        "What does TRL mean?",
        "How does risk management work?",
        "What is a Work Breakdown Structure?",
        "What happens during a Critical Design Review?",
        "How does verification relate to system design?",
    ]

    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state["question_input"] = q


# =============================================================================
# Guard: Ensure DB exists
# =============================================================================
if not collection_exists():
    st.error("Vector database is empty. Run:\n```bash\npython ingest.py\n```")
    st.stop()


# =============================================================================
# Input Section
# =============================================================================
question_input = st.text_input(
    "Your question:",
    placeholder="e.g. What is the Vee Model?",
    key="question_input"
)

ask_button = st.button("Ask", type="primary")


# =============================================================================
# Main QA Pipeline
# =============================================================================
if ask_button and question_input.strip():

    # Step 1: Query preprocessing
    clean_q = process_query(question_input)

    if clean_q.rstrip("?") != question_input.strip().rstrip("?"):
        st.info(f"Expanded query: **{clean_q}**")

    # Step 2: Retrieval
    with st.spinner("🔍 Searching handbook..."):
        chunks = retrieve(clean_q)

    # Step 3: Answer generation
    with st.spinner(f"🧠 Generating answer with {LLM_MODEL}..."):
        result = generate_answer(clean_q, chunks)


    # =============================================================================
    # Display Answer
    # =============================================================================
    st.subheader("📘 Answer")
    st.write(result["answer"])


    # =============================================================================
    # Citations
    # =============================================================================
    col1, col2 = st.columns(2)

    with col1:
        if result["source_pages"]:
            pages_str = ", ".join(str(p) for p in result["source_pages"])
            st.info(f"📄 Pages: {pages_str}")

    with col2:
        if result["source_sections"]:
            secs_str = ", ".join(str(s) for s in result["source_sections"])
            st.info(f"📚 Sections: {secs_str}")


    # =============================================================================
    # Context Visualization
    # =============================================================================
    direct_chunks = [
        c for c in result["chunks_used"]
        if c.get("source") != "cross_reference"
    ]

    xref_chunks = [
        c for c in result["chunks_used"]
        if c.get("source") == "cross_reference"
    ]


    # ---- Direct chunks ----
    with st.expander(f"📦 Retrieved context ({len(direct_chunks)} direct chunks)"):

        for i, chunk in enumerate(direct_chunks):

            label = f"Chunk {i+1} — Page {chunk['page_number']}"

            if chunk.get("section_number"):
                label += f" | Section {chunk['section_number']}"

            if chunk.get("section_title"):
                label += f" — {chunk['section_title']}"

            label += f" (distance: {chunk['distance']})"

            st.markdown(f"**{label}**")

            st.text(
                chunk["text"][:500] +
                ("..." if len(chunk["text"]) > 500 else "")
            )

            st.divider()


    # ---- Cross-referenced chunks ----
    if xref_chunks:

        with st.expander(f"🔗 Cross-referenced sections ({len(xref_chunks)} chunks)"):
            st.caption(
                "These sections were automatically fetched because "
                "other chunks referenced them."
            )

            for i, chunk in enumerate(xref_chunks):

                sec_info = (
                    f"Section {chunk['section_number']}"
                    if chunk.get("section_number") else ""
                )

                st.markdown(
                    f"**Cross-ref {i+1} — Page {chunk['page_number']} | {sec_info}**"
                )

                st.text(
                    chunk["text"][:400] +
                    ("..." if len(chunk["text"]) > 400 else "")
                )

                st.divider()


elif ask_button:
    st.warning("Please type a question first.")


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    "🚀 NASA Handbook QA — Streamlit + ChromaDB + Sentence Transformers + Ollama\n"
    
)