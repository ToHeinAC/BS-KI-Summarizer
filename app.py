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
from langchain_core.callbacks import BaseCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_summarizer_app")


# Custom callback handler for streaming tokens to Streamlit
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens to Streamlit."""
    
    def __init__(self, container):
        """Initialize with a Streamlit container."""
        super().__init__()
        self.container = container
        self.text = ""
        self.text_placeholder = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running."""
        if self.text_placeholder is None:
            self.text_placeholder = self.container.empty()
    
    def on_llm_new_token(self, token, **kwargs):
        """Run on new LLM token."""
        self.text += token
        # Display last 3 lines of text
        lines = self.text.split('\n')
        display_text = '\n'.join(lines[-3:]) if len(lines) > 3 else self.text
        self.text_placeholder.markdown(f"**Latest tokens:**\n```\n{display_text}\n```")
    
    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        pass
    
    def on_llm_error(self, error, **kwargs):
        """Run when LLM errors."""
        self.container.error(f"Error: {error}")
        
    def get_text(self):
        """Get the accumulated text."""
        return self.text

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
    ["llama3.3:70b-instruct-q4_K_M", "qwen3:30b-a3b", "llama3.3", "qwq"],
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
                
                # Create a container for streaming output
                stream_container = st.empty()
                stream_container.info("Generating summary... This may take a while for large documents.")
                
                # Start time for performance tracking
                start_time = time.time()
                
                try:
                    # Create our Streamlit handler
                    streamlit_handler = StreamlitCallbackHandler(stream_container)
                    
                    # Import the necessary components for callback management
                    from langchain_core.callbacks import CallbackManager
                    
                    # Check if the LLM has a callback_manager attribute
                    if hasattr(summarizer.final_llm, 'callback_manager'):
                        # If callback_manager is None, create a new one
                        if summarizer.final_llm.callback_manager is None:
                            summarizer.final_llm.callback_manager = CallbackManager(handlers=[streamlit_handler])
                        else:
                            # Add our handler to the existing callback manager's handlers list
                            if hasattr(summarizer.final_llm.callback_manager, 'handlers'):
                                summarizer.final_llm.callback_manager.handlers.append(streamlit_handler)
                            else:
                                # Create a new handlers list if it doesn't exist
                                summarizer.final_llm.callback_manager.handlers = [streamlit_handler]
                                
                        st.info("Streaming tokens enabled. You'll see the last 3 lines of generated text in real-time.")
                    else:
                        st.warning("Could not set up streaming tokens: LLM model doesn't support callbacks")
                except Exception as e:
                    st.warning(f"Could not set up streaming tokens: {e}")
                
                # Create status indicators
                status_container = st.empty()
                model_info = st.empty()
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Define a progress callback function
                def update_progress(current, total, step="chunk"):
                    progress = current / total
                    progress_bar.progress(progress)
                    
                    if step == "chunk":
                        status_container.info("📊 **Current Step:** Chunk Summarization")
                        model_info.success(f"🤖 **Using Model:** {summarizer.chunk_llm.model} for chunk summarization")
                        progress_text.text(f"Processing chunk {current} of {total} ({int(progress * 100)}%)")
                    elif step == "final":
                        status_container.info("📑 **Current Step:** Final Report Compilation")
                        model_info.success(f"🤖 **Using Model:** {summarizer.final_llm.model} for final summary")
                        progress_text.text(f"Generating final summary... ({int(progress * 100)}% complete)")
                
                # Update the document_summarizer.py to handle step tracking
                # Modify the summarize method to pass step information to our app
                
                # Start with chunk summarization step
                update_progress(0, 1, step="chunk")
                
                # Summarize document with progress tracking
                result = summarizer.summarize(
                    documents, 
                    progress_callback=update_progress,
                    on_final_start=lambda: update_progress(0, 1, step="final"),
                    on_final_complete=lambda: update_progress(1, 1, step="final")
                )
                
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
                
                # Create a container for the download button
                download_col1, download_col2 = st.columns([1, 3])
                
                # Prepare the summary text for download
                summary_text = result["summary"]
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"summary_{uploaded_file.name.split('.')[0]}_{timestamp}.txt"
                
                # Create a more prominent download button
                with download_col1:
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary_text,
                        file_name=filename,
                        mime="text/plain",
                        help="Download the generated summary as a text file",
                        use_container_width=True,
                    )
                
                with download_col2:
                    st.info(f"Download the complete summary as a text file: {filename}")
                    
                # Add another download option for full report (including intermediate summaries)
                if save_intermediate and result["intermediate_summaries"]:
                    full_report = summary_text + "\n\n" + "=" * 80 + "\n"
                    full_report += "INTERMEDIATE CHUNK SUMMARIES:\n" + "=" * 80 + "\n\n"
                    
                    for i, chunk_summary in enumerate(result["intermediate_summaries"]):
                        full_report += f"CHUNK {i+1}:\n" + "-" * 40 + "\n"
                        full_report += chunk_summary + "\n\n"
                    
                    report_filename = f"full_report_{uploaded_file.name.split('.')[0]}_{timestamp}.txt"
                    
                    st.download_button(
                        label="📥 Download Full Report (with Intermediate Summaries)",
                        data=full_report,
                        file_name=report_filename,
                        mime="text/plain",
                        help="Download a complete report including the final summary and all intermediate chunk summaries",
                        use_container_width=True,
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
