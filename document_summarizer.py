#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document Summarizer

A tool for summarizing documents using LangChain and LLMs.
Supports both small documents (stuff method) and large documents (map-reduce method).
Includes verbose output to track the summarization process.
"""

import os
import time
import logging
import argparse
from typing import Dict, List, Any, Optional

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_summarizer")


class DocumentSummarizer:
    """A class to handle document summarization with different strategies."""

    def __init__(
        self,
        model_name: str = "llama3.3",
        chunk_llm_name: str = "llama3.2",  # Model for chunk summarization
        chunk_size: int = 4000,
        chunk_overlap: int = 500,
        verbose: bool = True,
        language: str = "English"
    ):
        """Initialize the DocumentSummarizer.
        
        Args:
            model_name: Name of the LLM model to use
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            verbose: Whether to show verbose output
            language: Language for summarization prompts
        """
        self.language = language
        self.verbose = verbose
        
        # Initialize callback manager for verbose output
        callbacks = [StreamingStdOutCallbackHandler()] if verbose else []
        callback_manager = CallbackManager(handlers=callbacks)
        
        # Initialize LLMs
        if self.verbose:
            logger.info(f"Initializing final LLM with model: {model_name}")
            logger.info(f"Initializing chunk LLM with model: {chunk_llm_name}")
            
        # Main LLM for final summarization (with larger context window)
        self.final_llm = Ollama(
            model=model_name,
            callback_manager=callback_manager
        )
        
        # Chunk LLM for summarizing individual chunks
        self.chunk_llm = Ollama(
            model=chunk_llm_name,
            callback_manager=callback_manager
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Define summarization templates
        self.summarize_template = """
        You are a master senior analyst. 
        Your task is to create a deep research report for the information given to you, highlighting the most critical insights for decision-makers. 
        In order to do this, YOU MUST take the full information provided to you. 
        This can be either a full document (if the size is not too large) or a collection of ordered document chunks representing a full document (if the size indeed is large). 
        For the case of ordered document chunks, the order is important to recognize for providing the deep research report in the sense that the order marks a consecutive story line.
        
        KEY POINTS:
        YOU MUST respond in language: {language}
        Your deep research report should be at least 5000 words.
        Answer in correct professional terminology and sociolect maintaining exact key terms.

        Follow this specific structure:
        1. Introduction (approximately 20% of summary)
           - Provide context and background
           - State the document's purpose and significance
           
        2. Main Content (approximately 60% of summary)
           - Present key points and findings
           - Include important data, research, or evidence
           - Maintain the logical flow of the original document
           
        3. Conclusion (approximately 20% of summary)
           - Summarize main takeaways
           - Highlight implications or recommendations

        Document to summarize:
        {text}

        Key RULES: 
        - Be precise and comprehensive, focusing on the most important information.
        - Maintain exact figures, data points, sections and paragraphs as much as possible.

        Return your deep research report without any prefix or suffix to the summary, just your summary without any thinking passages.
        """
        
        self.map_template = """You are a senior analyst helping the master senior analyst to compile the final summary of a document. 
        In order to help the master senior analyst to compile the final summary of a document, YOU MUST summarize this section of the document in a few paragraphs maintaining all key information. 
        MOST IMPORTANT:
        - Summarize such that the connection to previous document chunks is taken into account.
        - Summarize such that you REDUCE the words count of the original document which is a chunk of a larger document BY A FACTOR OF 3 TO 5.
        - NEVER add any additional information or context that was not present in the original document.
        - YOU MUST maintain exact figures, data points, sections and paragraphs.
        - YOU MUST maintain the logical flow and arguments of the original document.
        - YOU MUST respond in language: {language}

        RULES: 
        - Answer in correct professional terminology and sociolect maintaining exact key terms.
        - Be precise and comprehensive, focusing on the most important information.
        - Maintain exact figures, data points, sections and paragraphs as much as possible.

        Return your summary without any prefix or suffix to the summary, just your summary without any thinking passages.
        
        Chunk to summarize:
        {text}
        """
        
        # Initialize chains
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize the summarization chains."""
        if self.verbose:
            logger.info("Initializing summarization chains")
        
        # Chain for small documents (stuff method)
        self.stuff_chain = load_summarize_chain(
            self.final_llm,  # Use the main LLM for stuff chain
            chain_type="stuff",
            prompt=PromptTemplate(
                input_variables=["text", "language"],
                template=self.summarize_template
            ),
            verbose=self.verbose
        )
        
        # Chain for large documents (map-reduce method)
        # Note: This is kept for compatibility but we'll use a custom implementation
        # that allows different LLMs for map and reduce steps
        self.map_reduce_chain = load_summarize_chain(
            self.final_llm,
            chain_type="map_reduce",
            map_prompt=PromptTemplate(
                input_variables=["text", "language"],
                template=self.map_template
            ),
            combine_prompt=PromptTemplate(
                input_variables=["text", "language"],
                template=self.summarize_template
            ),
            verbose=self.verbose
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        if self.verbose:
            logger.info(f"Loading document from {file_path}")
        
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.txt', '.md', '.html')):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        docs = loader.load()
        if self.verbose:
            logger.info(f"Loaded {len(docs)} pages/sections from document")
        
        return docs
    
    def summarize(self, documents: List[Document], progress_callback=None, on_final_start=None, on_final_complete=None) -> Dict[str, Any]:
        """Summarize the provided documents.
        
        Args:
            documents: List of Document objects to summarize
            progress_callback: Optional callback function to report progress during map-reduce
            on_final_start: Optional callback function to notify when final summary starts
            on_final_complete: Optional callback function to notify when final summary completes
            
        Returns:
            Dictionary containing the final summary and intermediate results if available
        """
        start_time = time.time()
        
        # Calculate total document size
        total_chars = sum(len(doc.page_content) for doc in documents)
        if self.verbose:
            logger.info(f"Document size: {total_chars} characters")
        
        # Choose appropriate chain based on document size
        if total_chars < 10000:  # Small document threshold
            if self.verbose:
                logger.info("Document is small. Using 'stuff' chain.")
            result = self.stuff_chain.invoke({
                "input_documents": documents,
                "language": self.language
            })
            
            return {
                "summary": result["output_text"],
                "method": "stuff",
                "intermediate_summaries": None,
                "chunks": None
            }
        else:
            if self.verbose:
                logger.info(f"Document is large ({total_chars} chars). Using map_reduce chain.")
                logger.info("Splitting document into chunks...")
            
            # Split documents for map-reduce
            split_docs = self.text_splitter.split_documents(documents)
            
            if self.verbose:
                logger.info(f"Split into {len(split_docs)} chunks")
                logger.info("Starting map-reduce summarization...")
            
            # Create map chain to access intermediate results
            from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
            from langchain.chains.combine_documents.stuff import StuffDocumentsChain
            from langchain.chains.llm import LLMChain
            
            # Create map chain with chunk_llm (lighter model for individual chunks)
            if self.verbose:
                logger.info(f"Using {self.chunk_llm.model} for chunk summarization")
                
            map_chain = LLMChain(
                llm=self.chunk_llm,  # Use chunk LLM for summarizing individual chunks
                prompt=PromptTemplate(
                    input_variables=["text", "language"],
                    template=self.map_template
                )
            )
            
            # Create combine chain with final_llm (stronger model for final summary)
            if self.verbose:
                logger.info(f"Using {self.final_llm.model} for final combined summary")
                
            combine_chain = LLMChain(
                llm=self.final_llm,  # Use final LLM for combining summaries
                prompt=PromptTemplate(
                    input_variables=["text", "language"],
                    template=self.summarize_template
                )
            )
            
            # Create map_reduce chain
            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                combine_document_chain=StuffDocumentsChain(
                    llm_chain=combine_chain,
                    document_variable_name="text"
                ),
                document_variable_name="text",
                return_intermediate_steps=True  # This is key to get intermediate summaries
            )
            
            # If we have a progress callback, we'll need to track progress manually
            if progress_callback:
                total_chunks = len(split_docs)
                processed_chunks = 0
                
                # Define a custom map function that updates progress
                def map_with_progress(docs):
                    nonlocal processed_chunks
                    results = []
                    
                    for doc in docs:
                        # Process the document with the map chain
                        result = map_chain.invoke({
                            "text": doc.page_content,
                            "language": self.language
                        })["text"]
                        
                        # Update progress
                        processed_chunks += 1
                        progress_callback(processed_chunks, total_chunks)
                        
                        # Store the result
                        results.append(result)
                    
                    return results
                
                # Process documents with progress tracking
                intermediate_steps = map_with_progress(split_docs)
                
                # Notify that we're starting the final summary step
                if on_final_start:
                    on_final_start()
                    
                # Combine the results
                combined_docs = [Document(page_content=text) for text in intermediate_steps]
                final_result = combine_chain.invoke({
                    "text": "\n\n".join([doc.page_content for doc in combined_docs]),
                    "language": self.language
                })["text"]
                
                # Notify that we've completed the final summary step
                if on_final_complete:
                    on_final_complete()
                
                # Create a result object similar to what map_reduce_chain would return
                result = {
                    "output_text": final_result,
                    "intermediate_steps": intermediate_steps
                }
            else:
                # Process with standard map-reduce if no progress callback
                # But still notify about the final summary step if callbacks are provided
                
                # First run the map step to get intermediate summaries
                intermediate_steps = []
                for doc in split_docs:
                    result = map_chain.invoke({
                        "text": doc.page_content,
                        "language": self.language
                    })["text"]
                    intermediate_steps.append(result)
                
                # Notify that we're starting the final summary step
                if on_final_start:
                    on_final_start()
                
                # Combine the results
                combined_docs = [Document(page_content=text) for text in intermediate_steps]
                final_result = combine_chain.invoke({
                    "text": "\n\n".join([doc.page_content for doc in combined_docs]),
                    "language": self.language
                })["text"]
                
                # Notify that we've completed the final summary step
                if on_final_complete:
                    on_final_complete()
                    
                # Create a result object similar to what map_reduce_chain would return
                result = {
                    "output_text": final_result,
                    "intermediate_steps": intermediate_steps
                }
            
            # Extract intermediate summaries
            intermediate_summaries = result.get("intermediate_steps", [])
            
            end_time = time.time()
            if self.verbose:
                logger.info(f"Summarization completed in {end_time - start_time:.2f} seconds")
                logger.info(f"Generated {len(intermediate_summaries)} intermediate summaries")
            
            return {
                "summary": result["output_text"],
                "method": "map_reduce",
                "intermediate_summaries": intermediate_summaries,
                "chunks": [doc.page_content for doc in split_docs]
            }


def main():
    """Main function to run the document summarizer from command line."""
    parser = argparse.ArgumentParser(description="Summarize documents using LLMs")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the document file to summarize"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3",
        help="LLM model to use for final summarization (default: qwen3)"
    )
    parser.add_argument(
        "--chunk-model", 
        type=str, 
        default="llama3.2",
        help="LLM model to use for chunk summarization (default: llama3.2)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=4000,
        help="Size of text chunks for splitting documents (default: 4000)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=500,
        help="Overlap between chunks (default: 500)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="English",
        help="Language for summarization (default: English)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path for the summary (optional)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Disable verbose output"
    )
    parser.add_argument(
        "--save-chunks", 
        action="store_true",
        help="Save the original chunks along with summaries"
    )
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = DocumentSummarizer(
        model_name=args.model,
        chunk_llm_name=args.chunk_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=not args.quiet,
        language=args.language
    )
    
    # Load and summarize document
    documents = summarizer.load_document(args.file_path)
    result = summarizer.summarize(documents)
    summary = result["summary"]
    
    # Output summary
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
            
            # Also save intermediate summaries if available
            if result["intermediate_summaries"]:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("INTERMEDIATE CHUNK SUMMARIES:\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk_summary in enumerate(result["intermediate_summaries"]):
                    f.write(f"CHUNK {i+1}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(chunk_summary + "\n\n")
                    
                    # Save original chunks if requested
                    if args.save_chunks and result["chunks"]:
                        f.write("ORIGINAL CHUNK CONTENT:\n")
                        f.write("-" * 40 + "\n")
                        if i < len(result["chunks"]):
                            f.write(result["chunks"][i] + "\n\n")
        
        print(f"\nSummary saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(summary)
        print("=" * 80)
        
        # Print intermediate summaries if available
        if result["method"] == "map_reduce" and result["intermediate_summaries"]:
            print("\n" + "=" * 80)
            print("INTERMEDIATE CHUNK SUMMARIES:")
            print("=" * 80)
            
            for i, chunk_summary in enumerate(result["intermediate_summaries"]):
                print(f"\nCHUNK {i+1}:")
                print("-" * 40)
                print(chunk_summary)


if __name__ == "__main__":
    main()
