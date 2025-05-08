#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document Summarizer Streamlit App

A web interface for summarizing documents using LangChain and LLMs.
Supports both small documents (stuff method) and large documents (map-reduce method).
"""

import os
import time
import logging
import streamlit as st
from typing import Dict, List, Any, Optional
from document_summarizer import DocumentSummarizer
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_summarizer_app")

# Set page configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Document Summarizer")
st.markdown(
    """Upload a document (PDF, TXT, MD, HTML) and get a comprehensive summary using LLMs.
    The app supports both small and large documents with different summarization strategies."""
)

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select LLM Model for Final Summary",
    ["llama3.3:70b-instruct-q4_K_M", "qwen3:30b-a3b", "llama3.3", "qwen3"],
    index=0,
    help="The LLM model to use for the final summary generation"
)

# Chunk model selection
chunk_llm_name = st.sidebar.selectbox(
    "Select LLM Model for Chunk Summarization",
    ["llama3.2", "llama3", "qwen2"],
    index=0,
    help="The LLM model to use for summarizing individual chunks"
)

# Language selection
language = st.sidebar.selectbox(
    "Summary Language",
    ["English", "German", "French", "Spanish", "Chinese", "Japanese"],
    index=0,
    help="The language for the summary output"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    chunk_size = st.slider(
        "Chunk Size", 
        min_value=1000, 
        max_value=20000, 
        value=10000, 
        step=1000,
        help="Size of text chunks for splitting documents"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap", 
        min_value=0, 
        max_value=2000, 
        value=1000, 
        step=100,
        help="Overlap between chunks"
    )
    
    save_intermediate = st.checkbox(
        "Show Intermediate Summaries", 
        value=True,
        help="Display intermediate chunk summaries"
    )
    
    show_chunks = st.checkbox(
        "Show Original Chunks", 
        value=False,
        help="Display original content chunks"
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["pdf", "txt", "md", "html"],
    help="Upload a PDF, TXT, MD, or HTML file to summarize"
)

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("files", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("files", uploaded_file.name)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Process button
if uploaded_file is not None:
    # Ensure files directory exists
    os.makedirs("files", exist_ok=True)
    
    # Save the uploaded file
    file_path = save_uploaded_file(uploaded_file)
    
    if file_path:
        if st.button("Summarize Document"):
            try:
                with st.spinner("Initializing summarizer..."):
                    # Initialize summarizer with selected options
                    summarizer = DocumentSummarizer(
                        model_name=model_name,
                        chunk_llm_name=chunk_llm_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        verbose=True,
                        language=language
                    )
                
                with st.spinner("Loading document..."):
                    # Load document
                    documents = summarizer.load_document(file_path)
                    st.info(f"Loaded {len(documents)} pages/sections from document")
                
                with st.spinner("Generating summary... This may take a while for large documents."):
                    # Start time for performance tracking
                    start_time = time.time()
                    
                    # Summarize document
                    result = summarizer.summarize(documents)
                    
                    # Calculate time taken
                    end_time = time.time()
                    time_taken = end_time - start_time
                
                # Display results
                st.success(f"Summary generated in {time_taken:.2f} seconds using {result['method']} method")
                
                # Display summary in a container
                st.subheader("Document Summary")
                st.markdown("---")
                st.markdown(result["summary"])
                st.markdown("---")
                
                # Display intermediate summaries if available and requested
                if save_intermediate and result["intermediate_summaries"]:
                    with st.expander("Intermediate Chunk Summaries"):
                        for i, chunk_summary in enumerate(result["intermediate_summaries"]):
                            st.markdown(f"**Chunk {i+1}**")
                            st.markdown(chunk_summary)
                            st.markdown("---")
                            
                            # Show original chunks if requested
                            if show_chunks and result["chunks"] and i < len(result["chunks"]):
                                with st.expander(f"Original Content - Chunk {i+1}"):
                                    st.text(result["chunks"][i])
                
                # Option to download summary
                summary_text = result["summary"]
                st.download_button(
                    label="Download Summary",
                    data=summary_text,
                    file_name=f"summary_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                logger.error(f"Summarization error: {e}", exc_info=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """**About**
    
    This app uses LangChain and LLMs to summarize documents.
    It supports different summarization strategies based on document size.
    
    - Small documents: 'stuff' method (all content at once)
    - Large documents: 'map-reduce' method (chunk by chunk)
    """
)
