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
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LangGraph imports
from langgraph.graph import StateGraph, END, START
# Remove unused imports that are causing issues with newer langgraph versions
from typing import List, Dict, Any, Sequence, Optional, Union, Callable

# Define state schema for LangGraph workflow
class DocumentAnalysis(TypedDict):
    """Schema for document analysis output"""
    title: str
    rationale: str
    structure: List[Dict[str, Any]]

class SummarizerState(TypedDict):
    """Schema for the summarizer workflow state"""
    # Input document(s)
    documents: List[Document]
    # Chunked documents (if using map-reduce)
    chunks: Optional[List[Document]]
    # Intermediate summaries from map step
    intermediate_summaries: Optional[List[str]]
    # Document analysis results
    analysis: Optional[DocumentAnalysis]
    # Final summary
    summary: Optional[str]
    # Method used (stuff or map_reduce)
    method: Optional[str]
    # Language for summarization
    language: str

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
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
        
        # Define summarization templatesBUP
        """Follow this specific structure:
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

        Return your deep research report without any prefix or suffix to the summary, just your summary without any thinking passages.""" 

        # Define summarization templates with separate system and human prompts
        self.summarize_system_template = """You are a master senior analyst. 
Your task is to create a deep, detailled (at least 5000 words) research report from information called DOCUMENT given to you, highlighting the most critical insights for decision-makers. 
In order to do this, YOU MUST analyse the full information provided to you. 
The DOCUMENT, i.e. the information given to you, can be either a full original document (if the size is not too large) or a collection of ordered document chunks representing a full original document (if the size indeed is large). 
For the case of ordered document chunks, the order is important to recognize for providing the deep research report as the order marks a consecutive story line.

KEY POINTS:
- YOU MUST respond in language: {language}
- Find out the title (full title of the DOCUMENT) and the rationale of the document and include it in your deep research report.
- Answer in correct professional terminology and sociolect maintaining exact key terms, figures and data points that underline your research."""
        
        self.summarize_human_template = """Here is the DOCUMENT for your in-depth research report (at least 5000 words):
{text}"""

        
        self.map_system_template = """You are a senior analyst helping the master senior analyst to compile the final summary of a document. 
In order to help the master senior analyst to compile the final summary of a document, YOU MUST summarize this section of the document in a few paragraphs maintaining all key information. 
MOST IMPORTANT:
- Summarize such that the connection to previous document chunks is taken into account.
- Summarize such that you REDUCE the words count of the original document which is a chunk of a larger document BY A FACTOR OF 3 TO 5.
- NEVER add any additional information or context that was not present in the original document.
- Maintain the logical flow and arguments of the original document.
- Maintain exact figures, data points, sections and paragraphs as much as possible.
- YOU MUST respond in language: {language}

RULES: 
- Answer in correct professional terminology and sociolect maintaining exact key terms.
- Be deep, precise and comprehensive, focusing on the most important information.

Return your summary without any prefix or suffix to the summary, just your summary without any thinking passages."""
        
        self.map_human_template = """Chunk to summarize:
{text}"""
        
        # Analysis step templates
        self.analysis_system_template = """You are a document analysis expert.

Your task is to analyze the provided document or document summaries and extract key information for structuring a deep research report.

You must perform two critical tasks:
1. Identify the full title of the document
2. Analyze the document's rationale and prepare a systematic structure for a deep report

YOUR ANALYSIS MUST BE IN {language}.

You must return your analysis as a valid JSON object with the following structure:
{
    "title": "The complete document title",
    "rationale": "A comprehensive analysis of the document's core purpose, context, and significance",
    "structure": [
        {
            "section": "Section name",
            "description": "What this section should cover",
            "subsections": [
                {
                    "subsection": "Subsection name",
                    "description": "What this subsection should cover"
                }
            ]
        }
    ]
}

The structure should be detailed and comprehensive, suitable for a deep research report of at least 5000 words.
"""
        
        self.analysis_human_template = """Document to analyze:
{text}

Provide a structured JSON analysis with the document title, rationale, and a detailed report structure."""
        
        # Initialize chains
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize the summarization chains."""
        if self.verbose:
            logger.info("Initializing summarization chains")
        
        # Chain for small documents (stuff method)
        summarize_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.summarize_system_template),
            HumanMessagePromptTemplate.from_template(self.summarize_human_template)
        ])
        
        self.stuff_chain = load_summarize_chain(
            self.final_llm,  # Use the main LLM for stuff chain
            chain_type="stuff",
            prompt=summarize_chat_prompt,
            verbose=self.verbose
        )
        
        # Chain for large documents (map-reduce method)
        # Note: This is kept for compatibility but we'll use a custom implementation
        # that allows different LLMs for map and reduce steps
        map_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.map_system_template),
            HumanMessagePromptTemplate.from_template(self.map_human_template)
        ])
        
        self.map_reduce_chain = load_summarize_chain(
            self.final_llm,
            chain_type="map_reduce",
            map_prompt=map_chat_prompt,
            combine_prompt=summarize_chat_prompt,
            verbose=self.verbose
        )
        
        # Analysis chain for document structure and title extraction
        analysis_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.analysis_system_template),
            HumanMessagePromptTemplate.from_template(self.analysis_human_template)
        ])
        
        self.analysis_chain = analysis_chat_prompt | self.final_llm
    
    def _map_step(self, state: SummarizerState) -> SummarizerState:
        """Map step: Process individual document chunks."""
        if self.verbose:
            logger.info("Starting map step for document chunks")
        
        # Get documents from state
        documents = state["documents"]
        language = state["language"]
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        if self.verbose:
            logger.info(f"Split into {len(split_docs)} chunks")
        
        # Process each chunk with map chain
        intermediate_summaries = []
        for doc in split_docs:
            response = self.chunk_llm.invoke(
                self.map_chat_prompt.format_messages(
                    text=doc.page_content,
                    language=language
                )
            )
            # Handle both string and AIMessage responses
            result = response.content if hasattr(response, 'content') else response
            intermediate_summaries.append(result)
        
        # Update state
        return {
            **state,
            "chunks": split_docs,
            "intermediate_summaries": intermediate_summaries
        }
    
    def _analysis_step(self, state: SummarizerState) -> SummarizerState:
        """Analysis step: Extract document title and prepare report structure."""
        if self.verbose:
            logger.info("Starting analysis step for document structure")
        
        language = state["language"]
        method = state["method"]
        
        # Determine what text to analyze based on method
        if method == "stuff":
            # For small documents, analyze the original text
            text_to_analyze = "\n\n".join([doc.page_content for doc in state["documents"]])
        else:
            # For map-reduce, analyze the intermediate summaries
            text_to_analyze = "\n\n".join(state["intermediate_summaries"])
        
        # Run analysis chain
        analysis_result = self.analysis_chain.invoke({
            "text": text_to_analyze,
            "language": language
        })
        
        # Parse JSON output
        try:
            # Extract JSON from potential text wrapper
            json_str = analysis_result
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "{" in json_str and "}" in json_str:
                # Find the first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}")+1
                json_str = json_str[start:end]
                
            analysis_data = json.loads(json_str)
            
            # Ensure required fields are present
            if not all(k in analysis_data for k in ["title", "rationale", "structure"]):
                raise ValueError("Missing required fields in analysis output")
                
            # Create DocumentAnalysis object
            analysis = DocumentAnalysis(
                title=analysis_data["title"],
                rationale=analysis_data["rationale"],
                structure=analysis_data["structure"]
            )
            
            if self.verbose:
                logger.info(f"Analysis complete. Document title: {analysis['title']}")
            
            # Update state
            return {**state, "analysis": analysis}
            
        except Exception as e:
            logger.error(f"Error parsing analysis result: {e}")
            logger.error(f"Raw analysis result: {analysis_result}")
            
            # Create a default analysis object
            analysis = DocumentAnalysis(
                title="Unknown Document Title",
                rationale="Unable to extract document rationale",
                structure=[{"section": "Main Content", "description": "Document content"}]
            )
            
            return {**state, "analysis": analysis}
    
    def _final_report_step(self, state: SummarizerState) -> SummarizerState:
        """Final report step: Generate the final report using analysis structure."""
        if self.verbose:
            logger.info("Starting final report generation")
        
        language = state["language"]
        method = state["method"]
        analysis = state["analysis"]
        
        # Create an enhanced prompt that incorporates the analysis
        enhanced_system_template = f"""{self.summarize_system_template}

IMPORTANT ADDITIONAL INSTRUCTIONS:
- The document title is: {analysis['title']}
- Document rationale: {analysis['rationale']}
- Follow this specific structure for your report:"""
        
        # Add the structure to the prompt
        for i, section in enumerate(analysis["structure"]):
            enhanced_system_template += f"\n{i+1}. {section['section']}: {section['description']}"
            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    enhanced_system_template += f"\n   {i+1}.{j+1}. {subsection['subsection']}: {subsection['description']}"
        
        # Create enhanced prompt
        enhanced_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(enhanced_system_template),
            HumanMessagePromptTemplate.from_template(self.summarize_human_template)
        ])
        
        # Generate final report based on method
        if method == "stuff":
            # For small documents
            text_to_summarize = "\n\n".join([doc.page_content for doc in state["documents"]])
        else:
            # For map-reduce, use intermediate summaries
            text_to_summarize = "\n\n".join(state["intermediate_summaries"])
        
        # Generate final report
        final_response = self.final_llm.invoke(
            enhanced_prompt.format_messages(
                text=text_to_summarize,
                language=language
            )
        )
        # Handle both string and AIMessage responses
        final_report = final_response.content if hasattr(final_response, 'content') else final_response
        
        # Update state
        return {**state, "summary": final_report}
    
    def _create_workflow(self) -> Any:
        """Create the LangGraph workflow for document summarization."""
        # Create workflow graph
        workflow = StateGraph(SummarizerState)
        
        # Add nodes
        workflow.add_node("map", self._map_step)
        workflow.add_node("analysis_node", self._analysis_step)
        workflow.add_node("final_report", self._final_report_step)
        
        # Add edges
        # For map-reduce method
        workflow.add_edge("map", "analysis_node")
        workflow.add_edge("analysis_node", "final_report")
        
        # For stuff method (skip map step)
        workflow.add_conditional_edges(
            START,
            lambda state: "analysis_node" if state["method"] == "stuff" else "map"
        )
        
        # End after final report
        workflow.add_edge("final_report", END)
        
        # Compile workflow
        return workflow.compile()
    
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
        """Summarize the provided documents using LangGraph workflow.
        
        Args:
            documents: List of Document objects to summarize
            progress_callback: Optional callback function to report progress during map-reduce
            on_final_start: Optional callback function to notify when final summary starts
            on_final_complete: Optional callback function to notify when final summary completes
            
        Returns:
            Dictionary containing the final summary and intermediate results if available
        """
        start_time = time.time()
        
        # Store callbacks for later use
        self.progress_callback = progress_callback
        self.on_final_start = on_final_start
        self.on_final_complete = on_final_complete
        
        # Calculate total document size
        total_chars = sum(len(doc.page_content) for doc in documents)
        if self.verbose:
            logger.info(f"Document size: {total_chars} characters")
        
        # Determine method based on document size
        method = "stuff" if total_chars < 10000 else "map_reduce"
        
        if self.verbose:
            logger.info(f"Using {method} method for document summarization")
        
        # Initialize map_chat_prompt if it doesn't exist yet
        if not hasattr(self, "map_chat_prompt"):
            self.map_chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.map_system_template),
                HumanMessagePromptTemplate.from_template(self.map_human_template)
            ])
        
        # Create LangGraph workflow if not already created
        if not hasattr(self, "workflow"):
            self.workflow = self._create_workflow()
        
        # Initialize state
        initial_state = SummarizerState(
            documents=documents,
            chunks=None,
            intermediate_summaries=None,
            analysis=None,
            summary=None,
            method=method,
            language=self.language
        )
        
        # Execute workflow
        if self.verbose:
            logger.info("Starting LangGraph workflow for document summarization")
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results
            summary = final_state["summary"]
            analysis = final_state["analysis"]
            chunks = final_state.get("chunks")
            intermediate_summaries = final_state.get("intermediate_summaries")
            
            end_time = time.time()
            if self.verbose:
                logger.info(f"Summarization completed in {end_time - start_time:.2f} seconds")
                if intermediate_summaries:
                    logger.info(f"Generated {len(intermediate_summaries)} intermediate summaries")
                logger.info(f"Document title identified: {analysis['title']}")
            
            # Notify that we've completed the final summary step
            if self.on_final_complete:
                self.on_final_complete()
            
            # Return results in the same format as before for compatibility
            return {
                "summary": summary,
                "method": method,
                "intermediate_summaries": intermediate_summaries,
                "chunks": [doc.page_content for doc in chunks] if chunks else None,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {e}")
            
            # Fall back to the old implementation
            if self.verbose:
                logger.info("Falling back to traditional implementation")
            
            if method == "stuff":
                result = self.stuff_chain.invoke({
                    "input_documents": documents,
                    "language": self.language
                })
                
                return {
                    "summary": result["output_text"],
                    "method": "stuff",
                    "intermediate_summaries": None,
                    "chunks": None,
                    "analysis": None
                }
            else:
                # Use the old map-reduce implementation
                # This is a simplified version of the original code
                split_docs = self.text_splitter.split_documents(documents)
                
                # Map step
                intermediate_steps = []
                for i, doc in enumerate(split_docs):
                    response = self.chunk_llm.invoke(
                        self.map_chat_prompt.format_messages(
                            text=doc.page_content,
                            language=self.language
                        )
                    )
                    # Handle both string and AIMessage responses
                    result = response.content if hasattr(response, 'content') else response
                    intermediate_steps.append(result)
                    
                    # Update progress if callback provided
                    if self.progress_callback:
                        self.progress_callback(i + 1, len(split_docs))
                
                # Notify that we're starting the final summary step
                if self.on_final_start:
                    self.on_final_start()
                
                # Combine step
                summarize_chat_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(self.summarize_system_template),
                    HumanMessagePromptTemplate.from_template(self.summarize_human_template)
                ])
                
                final_response = self.final_llm.invoke(
                    summarize_chat_prompt.format_messages(
                        text="\n\n".join(intermediate_steps),
                        language=self.language
                    )
                )
                # Handle both string and AIMessage responses
                final_result = final_response.content if hasattr(final_response, 'content') else final_response
                
                # Notify that we've completed the final summary step
                if self.on_final_complete:
                    self.on_final_complete()
                
                return {
                    "summary": final_result,
                    "method": "map_reduce",
                    "intermediate_summaries": intermediate_steps,
                    "chunks": [doc.page_content for doc in split_docs],
                    "analysis": None
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
        
        # Print document analysis if available
        if "analysis" in result and result["analysis"]:
            analysis = result["analysis"]
            print("\n" + "=" * 80)
            print("DOCUMENT ANALYSIS:")
            print("=" * 80)
            print(f"Title: {analysis['title']}")
            print("\nRationale:")
            print(analysis['rationale'])
            print("\nStructure:")
            for i, section in enumerate(analysis['structure']):
                print(f"\n{i+1}. {section['section']}: {section['description']}")
                if 'subsections' in section:
                    for j, subsection in enumerate(section['subsections']):
                        print(f"   {i+1}.{j+1}. {subsection['subsection']}: {subsection['description']}")
        
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
