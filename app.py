"""
DocuMind - Streamlit Web Interface

Interactive web application for document Q&A with multimodal support.
"""

import os
import streamlit as st
from pathlib import Path
import time

from documind import DocuMind


# Page configuration
st.set_page_config(
    page_title="DocuMind - Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .confidence-high { color: #10b981; }
    .confidence-medium { color: #f59e0b; }
    .confidence-low { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'documind' not in st.session_state:
    st.session_state.documind = DocuMind()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 DocuMind</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">'
        'Intelligent Document Q&A powered by Multimodal RAG</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Supports PDF, PNG, JPG files"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Save file temporarily
                    temp_path = Path(f"./temp/{uploaded_file.name}")
                    temp_path.parent.mkdir(exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        result = st.session_state.documind.load_document(str(temp_path))
                        st.session_state.documents.append({
                            "name": uploaded_file.name,
                            "id": result["document_id"],
                            "chunks": result["num_chunks"]
                        })
                        st.success(f"✅ Processed {result['num_chunks']} chunks!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Document list
        st.header("📚 Loaded Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                st.markdown(f"**{doc['name']}**")
                st.caption(f"{doc['chunks']} chunks")
        else:
            st.info("No documents loaded yet")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        top_k = st.slider("Number of sources", 1, 10, 5)
        
        if st.button("🗑️ Clear All"):
            st.session_state.documind.clear()
            st.session_state.documents = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Chat interface
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Sources"):
                        for source in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>Page {source.get("page", "N/A")}</strong> '
                                f'({source.get("type", "text")})<br>'
                                f'{source.get("preview", "")}</div>',
                                unsafe_allow_html=True
                            )
        
        # Question input
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            with st.chat_message("user"):
                st.write(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    
                    try:
                        response = st.session_state.documind.query(
                            question=question,
                            top_k=top_k
                        )
                        
                        elapsed = time.time() - start_time
                        
                        st.write(response.answer)
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for source in response.sources:
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<strong>Page {source.get("page", "N/A")}</strong> '
                                    f'({source.get("type", "text")})<br>'
                                    f'{source.get("preview", "")}</div>',
                                    unsafe_allow_html=True
                                )
                        
                        # Add to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "confidence": response.confidence
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("📊 Statistics")
        
        # Metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Documents", len(st.session_state.documents))
        st.markdown('</div>', unsafe_allow_html=True)
        
        total_chunks = sum(d["chunks"] for d in st.session_state.documents)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Chunks", total_chunks)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Questions Asked", len([
            m for m in st.session_state.chat_history 
            if m["role"] == "user"
        ]))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Last response confidence
        if st.session_state.chat_history:
            last_response = [
                m for m in st.session_state.chat_history 
                if m["role"] == "assistant"
            ]
            if last_response and "confidence" in last_response[-1]:
                conf = last_response[-1]["confidence"]
                conf_class = (
                    "confidence-high" if conf > 0.8 
                    else "confidence-medium" if conf > 0.6 
                    else "confidence-low"
                )
                st.markdown(
                    f'<div class="metric-card">'
                    f'<p style="margin:0; color:#6b7280;">Last Confidence</p>'
                    f'<p class="{conf_class}" style="font-size:2rem; margin:0;">'
                    f'{conf:.0%}</p></div>',
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
