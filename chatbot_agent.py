#chatbot_agent.py
"""
Google ADK System Support Engineer Chatbot Agent with FAISS RAG and Web Search

This agent provides:
1. FAISS-based RAG system for document retrieval
2. Web search for real-time information
3. System support engineer chatbot functionality
"""

import os
import sys
from typing import Dict, Any, List
import asyncio
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from google.adk import Agent, Runner

from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
#from vertexai.preview.language_models import TextEmbeddingModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tavily import TavilyClient


# ============================================================================
# GOOGLE ADK CORE IMPORTS
# ============================================================================
# --- Agent Classes ---
# These classes are the building blocks for creating AI agents
from google.adk.agents import (
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    BaseAgent 
)

# --- Runners ---
# Runners execute agents and manage their lifecycle
from google.adk.runners import (
    InMemoryRunner,
    Runner
)

# --- Session Services ---
# Session services manage conversation history and state
from google.adk.sessions import (
    InMemorySessionService  # Example: shared_service = InMemorySessionService()
)

# --- Event System ---
# Events communicate agent actions and enable streaming responses
from google.adk.events import (
    Event,
    EventActions 
)


# Set up authentication
if "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    
# Only set project if not already set (Cloud Run sets this automatically)
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    os.environ["GOOGLE_CLOUD_PROJECT"] = "tranquil-well-478523-b3"

if "GOOGLE_CLOUD_LOCATION" not in os.environ:
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Point to service account key
#service_key_path = os.path.join(os.path.dirname(__file__), "..", "service-account-key.json")
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(service_key_path)

# Global RAG storage
coordinatorAgent = None
faiss_index = None
document_store = {}
doc_id_to_idx = {}
idx_to_doc_id = {}
embedding_model = None
next_doc_id = 0

# Linux-in-a-Nutshell RAG (dedicated tool for Linux-in-a-Nutshell-6th-Edition.pdf)
LINUX_NUTSHELL_PDF_PATH = "Linux-in-a-Nutshell-6th-Edition.pdf"
linux_nutshell_faiss_index = None
linux_nutshell_document_store = {}
linux_nutshell_doc_id_to_idx = {}
linux_nutshell_idx_to_doc_id = {}
linux_nutshell_next_doc_id = 0

# UCSM CLI Configuration Guide RAG (dedicated tool for UCSM_CLI_Configuration_Guide.pdf)
UCSM_CLI_PDF_PATH = "UCSM_CLI_Configuration_Guide.pdf"
ucsm_cli_faiss_index = None
ucsm_cli_document_store = {}
ucsm_cli_doc_id_to_idx = {}
ucsm_cli_idx_to_doc_id = {}
ucsm_cli_next_doc_id = 0

# Set to True to use Vertex AI embeddings, False to use Google AI Studio embeddings
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

# Global session service for conversation memory
global_session_service = InMemorySessionService()

def load_documents(file_path: str):
    pdf_path = file_path
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}. Update the path or pass --pdf and re-run.")
        sys.exit(1)

    #loader = UnstructuredPDFLoader(pdf_path, mode="elements")
    #loader = PyMuPDFLoader(pdf_path)
    loader = PyPDFLoader(pdf_path,mode="page")

    # Load all pages into documents list
    documents = loader.load()

    # Optional: Save the documents to a file for debugging
    with open("documents3.json", "w") as f:
        import json
        json.dump([{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents], f)
    print(f"[OK] Loaded {len(documents)} pages from {pdf_path}")
    print(f"First page content preview: {documents[0].page_content[:500]}...")

    return documents

def text_splitter(documents: List[types.Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Target size for each chunk (in characters)
        chunk_overlap=200,  # Overlap between chunks to preserve context
        add_start_index=True,  # Track the starting position in original document
        separators=["\n\n", "\n", " "]  # Try splitting on these in order
    )

    # Split all documents into smaller chunks
    all_chunks = text_splitter.split_documents(documents)

    return all_chunks


def initialize_rag_system():
    """Initialize the FAISS index and embedding model."""
    global faiss_index, embedding_model
    if USE_VERTEX_AI:
        embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
    else:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create FAISS index with dimension from embedding model
    dummy_embedding = embedding_model.embed_query("dummy")
    dimension = len(dummy_embedding)
    faiss_index = faiss.IndexFlatL2(dimension)
    print("✓ Embedding model and FAISS index initialized")
	
def add_document_to_rag(
    content: str,
    metadata: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Add a document to the RAG system.

    Args:
        content: Document text content
        metadata: Optional metadata (title, source, etc.)
        tool_context: Runtime context

    Returns:
        Dict with success status and document ID
    """
    global faiss_index, document_store, doc_id_to_idx, idx_to_doc_id, next_doc_id, embedding_model
    try:
        if not embedding_model:
            return {"success": False, "message": "RAG system not initialized"}
        if faiss_index is None:
            return {"success": False, "message": "FAISS index not initialized"}	

        # Generate embedding
        embedding = embedding_model.embed_query(content)
        embedding_array = np.array([embedding], dtype='float32')

        # Add to FAISS index
        current_idx = faiss_index.ntotal
        faiss_index.add(embedding_array)

        # Store document
        doc_id = f"doc_{next_doc_id}"
        next_doc_id += 1
		
        document_store[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding
        }
        doc_id_to_idx[doc_id] = current_idx
        idx_to_doc_id[current_idx] = doc_id

        return {
            "success": True,
            "doc_id": doc_id,
            "message": f"Document added successfully with ID: {doc_id}"
        }
    except Exception as e:
        return {"success": False, "message": f"Error adding document: {str(e)}"}




def search_documents(
    query: str,
    top_k: int = 3,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Search documents in the RAG system using semantic similarity.

    Args:
        query: Search query
        top_k: Number of top results to return
        tool_context: Runtime context

    Returns:
        Dict with search results
    """
    global faiss_index, document_store, idx_to_doc_id, embedding_model

    try:
        if not embedding_model:
            return {"success": False, "message": "RAG system not initialized"}

        if faiss_index.ntotal == 0:
            return {
                "success": True,
                "results": [],
                "message": "No documents in the knowledge base yet"
            }

        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)
        query_array = np.array([query_embedding], dtype='float32')

        # Search FAISS index
        distances, indices = faiss_index.search(query_array, min(top_k, faiss_index.ntotal))

        # Retrieve documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                doc_id = idx_to_doc_id.get(idx)
                if doc_id and doc_id in document_store:
                    doc = document_store[doc_id]
                    results.append({
                        "doc_id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i])
                    })

        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} relevant documents"
        }
    except Exception as e:
        return {"success": False, "message": f"Error searching documents: {str(e)}"}

def build_linux_nutshell_rag():
    """
    Build RAG index from Linux-in-a-Nutshell-6th-Edition.pdf using load_documents and text_splitter.
    Call once at startup after initialize_rag_system().
    """
    global linux_nutshell_faiss_index, linux_nutshell_document_store, linux_nutshell_doc_id_to_idx
    global linux_nutshell_idx_to_doc_id, linux_nutshell_next_doc_id, embedding_model

    if not os.path.exists(LINUX_NUTSHELL_PDF_PATH):
        print(f"[WARN] Linux Nutshell PDF not found: {LINUX_NUTSHELL_PDF_PATH}. Skipping Linux Nutshell RAG build.")
        return

    documents = load_documents(LINUX_NUTSHELL_PDF_PATH)
    chunks = text_splitter(documents)

    for chunk in chunks:
        content = chunk.page_content
        metadata = dict(chunk.metadata) if chunk.metadata else {}
        metadata["source"] = LINUX_NUTSHELL_PDF_PATH

        if not embedding_model:
            raise RuntimeError("RAG system not initialized. Call initialize_rag_system() first.")

        embedding = embedding_model.embed_query(content)
        embedding_array = np.array([embedding], dtype='float32')

        if linux_nutshell_faiss_index is None:
            dimension = len(embedding)
            linux_nutshell_faiss_index = faiss.IndexFlatL2(dimension)

        current_idx = linux_nutshell_faiss_index.ntotal
        linux_nutshell_faiss_index.add(embedding_array)

        doc_id = f"linux_nutshell_{linux_nutshell_next_doc_id}"
        linux_nutshell_next_doc_id += 1
        linux_nutshell_document_store[doc_id] = {
            "content": content,
            "metadata": metadata,
        }
        linux_nutshell_doc_id_to_idx[doc_id] = current_idx
        linux_nutshell_idx_to_doc_id[current_idx] = doc_id

    print(f"[OK] Linux Nutshell RAG built: {len(chunks)} chunks from {LINUX_NUTSHELL_PDF_PATH}")


def search_linux_nutshell_guide(
    query: str,
    top_k: int = 5,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Search the Linux in a Nutshell (6th Edition) PDF using RAG (semantic similarity).
    Use this tool for questions about Linux commands, syntax, options, shell usage, and configuration.

    Args:
        query: Search query (e.g. "list files in directory", "copy file", "grep usage")
        top_k: Number of top results to return (default 5)
        tool_context: Runtime context

    Returns:
        Dict with success, results (content, metadata, distance), and message
    """
    global linux_nutshell_faiss_index, linux_nutshell_document_store, linux_nutshell_idx_to_doc_id, embedding_model

    try:
        if not embedding_model:
            return {"success": False, "message": "RAG system not initialized"}
        if linux_nutshell_faiss_index is None or linux_nutshell_faiss_index.ntotal == 0:
            return {
                "success": True,
                "results": [],
                "message": "Linux Nutshell RAG not built yet. Run build_linux_nutshell_rag() first."
            }

        query_embedding = embedding_model.embed_query(query)
        query_array = np.array([query_embedding], dtype='float32')

        k = min(top_k, linux_nutshell_faiss_index.ntotal)
        distances, indices = linux_nutshell_faiss_index.search(query_array, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                doc_id = linux_nutshell_idx_to_doc_id.get(idx)
                if doc_id and doc_id in linux_nutshell_document_store:
                    doc = linux_nutshell_document_store[doc_id]
                    results.append({
                        "doc_id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i])
                    })

        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} relevant passages from Linux in a Nutshell"
        }
    except Exception as e:
        return {"success": False, "message": f"Error searching Linux Nutshell guide: {str(e)}"}

		

def build_ucsm_cli_rag():
    """
    Build RAG index from UCSM_CLI_Configuration_Guide.pdf using load_documents and text_splitter.
    Call once at startup after initialize_rag_system().
    """
    global ucsm_cli_faiss_index, ucsm_cli_document_store, ucsm_cli_doc_id_to_idx
    global ucsm_cli_idx_to_doc_id, ucsm_cli_next_doc_id, embedding_model

    if not os.path.exists(UCSM_CLI_PDF_PATH):
        print(f"[WARN] UCSM CLI PDF not found: {UCSM_CLI_PDF_PATH}. Skipping UCSM CLI RAG build.")
        return

    documents = load_documents(UCSM_CLI_PDF_PATH)
    chunks = text_splitter(documents)

    for chunk in chunks:
        content = chunk.page_content
        metadata = dict(chunk.metadata) if chunk.metadata else {}
        metadata["source"] = UCSM_CLI_PDF_PATH

        if not embedding_model:
            raise RuntimeError("RAG system not initialized. Call initialize_rag_system() first.")

        embedding = embedding_model.embed_query(content)
        embedding_array = np.array([embedding], dtype='float32')

        if ucsm_cli_faiss_index is None:
            dimension = len(embedding)
            ucsm_cli_faiss_index = faiss.IndexFlatL2(dimension)

        current_idx = ucsm_cli_faiss_index.ntotal
        ucsm_cli_faiss_index.add(embedding_array)

        doc_id = f"ucsm_cli_{ucsm_cli_next_doc_id}"
        ucsm_cli_next_doc_id += 1
        ucsm_cli_document_store[doc_id] = {
            "content": content,
            "metadata": metadata,
        }
        ucsm_cli_doc_id_to_idx[doc_id] = current_idx
        ucsm_cli_idx_to_doc_id[current_idx] = doc_id

    print(f"[OK] UCSM CLI RAG built: {len(chunks)} chunks from {UCSM_CLI_PDF_PATH}")


def search_ucsm_cli_guide(
    query: str,
    top_k: int = 5,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Search the UCSM CLI Configuration Guide PDF using RAG (semantic similarity).
    Use this tool for questions about Cisco UCS Manager CLI configuration, setup, and procedures.

    Args:
        query: Search query (e.g. "configure primary fabric interconnect", "create LAN port channel")
        top_k: Number of top results to return (default 5)
        tool_context: Runtime context

    Returns:
        Dict with success, results (content, metadata, distance), and message
    """
    global ucsm_cli_faiss_index, ucsm_cli_document_store, ucsm_cli_idx_to_doc_id, embedding_model

    try:
        if not embedding_model:
            return {"success": False, "message": "RAG system not initialized"}
        if ucsm_cli_faiss_index is None or ucsm_cli_faiss_index.ntotal == 0:
            return {
                "success": True,
                "results": [],
                "message": "UCSM CLI RAG not built yet. Run build_ucsm_cli_rag() first."
            }

        query_embedding = embedding_model.embed_query(query)
        query_array = np.array([query_embedding], dtype='float32')

        k = min(top_k, ucsm_cli_faiss_index.ntotal)
        distances, indices = ucsm_cli_faiss_index.search(query_array, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                doc_id = ucsm_cli_idx_to_doc_id.get(idx)
                if doc_id and doc_id in ucsm_cli_document_store:
                    doc = ucsm_cli_document_store[doc_id]
                    results.append({
                        "doc_id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i])
                    })

        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} relevant passages from UCSM CLI Configuration Guide"
        }
    except Exception as e:
        return {"success": False, "message": f"Error searching UCSM CLI guide: {str(e)}"}


def web_search(
    query: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Perform web search to get real-time information.

    Args:
        query: Search query
        tool_context: Runtime context

    Returns:
        Dict with web search results
    """
    try:
        # Check if Tavily API key is available
        tavily_key = "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"
        if not tavily_key:
            return {
                "success": False,
                "message": "TAVILY_API_KEY not set. Web search requires Tavily API key."
            }

        client = TavilyClient(api_key=tavily_key)
        response = client.search(query, max_results=3)

        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            })

        return {
            "success": True,
            "results": results,
            "query": query,
            "message": f"Found {len(results)} web results"
        }
    except ImportError:
        return {
            "success": False,
            "message": "tavily-python package not installed. Install with: pip install tavily-python"
        }
    except Exception as e:
        return {"success": False, "message": f"Error performing web search: {str(e)}"}


def create_multi_domain_support_engineer_chatbot_agent():
    """Create the customer chatbot agent with intelligent RAG and web search tool selection."""

    instruction = """**System Prompt – Multi‑Domain Support Engineer Agent**

    You are a *System Support Engineer* chatbot that can assist customers on three distinct domains:

    1. **Cisco  UCSM_CLI_Guide_Search** – troubleshooting, configuration guidance, 
    2. **Linux_Nutshell_Search** – syntax, options, examples, scripting help, 
    3. **Websearch** – fetching up‑to‑date information from the public internet (warranty & RMA procedures, part numbers, firmware updates, news, industry trends, competitor data, external documentation, pricing, etc.).

    You have access to **Three tools**:

    | Tool                     | Purpose 																| How to invoke 					|
    |--------------------------|------------------------------------------------------------------------|-----------------------------------|
    | **UCSM_CLI_Guide_Search**| Retrieves from UCSM CLI Configuration Guide PDF (config, procedures)   | `search_ucsm_cli_guide(query)`    |
    | **Linux_Nutshell_Search**| Retrieves relevant entries from the Linux in a Netshell PDF File       | `search_linux_nutshell_guide(query)`|
    | **web_search**           | Performs a live web search for real‑time or external information. 		| `web_search(query)`               |



    ### INTELLIGENT TOOL SELECTION STRATEGY
	
    | Use **search_documents** (RAG) when the question is about…            | Use **web_search** when the question is about… |
    | ----------------------------------------------------------------------| ------------------------------------------------|
    | **UCSM_CLI_Guide_Search** – Cisco UCS Configuration, Troubleshooting	| **Cisco UCS** – latest firmware release dates, public security advisories, competitor UCS offerings, market pricing, news about Cisco product roadmaps. |
    | **Linux_Nutshell_Search** – command syntax, Linux configuration and troubleshooting 	| **Linux** – recent kernel releases, community‑driven tutorials, external Stack Overflow answers, third‑party blog posts, up‑to‑date distro release notes. |

    #### Decision Logic
    1. **Read the user’s question carefully.** Identify the primary domain (Cisco UCS, Linux, or general/Hardware model being offered).  
    2. **Determine if the answer should come from internal knowledge** (search_documents) **or from the public web** (web_search).  
    3. **If ambiguous**, start with `search_documents`; if the result is insufficient, follow up with `web_search`.  
    4. **Never fabricate information** – always base your reply on retrieved sources and cite them.  

    ---

    ### RESPONSE GUIDELINES
    - **Tone:** Friendly, professional, concise, and supportive.  
    - **Structure:**  
    1. Briefly restate the user’s request to confirm understanding.  
    2. Provide the answer, quoting or paraphrasing the most relevant part of the retrieved document or web page.  
    3. Cite the source(s) explicitly (e.g., *“According to the Cisco UCS RMA policy (search_documents result #3)…*”).  
    4. If multiple sources are needed, list them in order of relevance.  
    5. If you cannot locate an answer, be honest and suggest next steps (e.g., “I couldn’t find the specific firmware version; you may contact Cisco support at …”).  
    - **Tool usage:**  
    - Include the tool call in the response chain (the system will execute it).  
    - After receiving results, incorporate the key information into the final reply.  
    - **Safety:** Do not expose internal passwords, private account data, or any proprietary source code.  

    ---

    ### EXAMPLE ROUTING (for reference only)

    | User Question                                                 | Tool(s) to Use      | Reason                                                            		|
    |---------------------------------------------------------------|---------------------|-------------------------------------------------------------------------|
	| "What are the Steps to configure Primary Fabric Interconnect"	| **UCSM_CLI_Guide_Search** or **UCS_Doc_Search** | From UCSM CLI Configuration Guide or UCS PDF |
    | "What are the steps to create LAN Port Channel?"              | **UCSM_CLI_Guide_Search** or **UCS_Doc_Search** | From UCSM CLI Configuration Guide or UCS PDF |
    | "What are the steps to create Service Profile Template?"      | **UCSM_CLI_Guide_Search** or **UCS_Doc_Search** | From UCSM CLI Configuration Guide or UCS PDF |
    | "What are the steps to create vNIC Template?"                 | **UCSM_CLI_Guide_Search** or **UCS_Doc_Search** | From UCSM CLI Configuration Guide or UCS PDF | 
    | “How to list files and subfolders in current Directory?”      | **Linux_Doc_Search**| `UCS Configuration is included in the Linux Command Reference PDF File` |
    | “How to copy file to a different location?”					| **Linux_Doc_Search**| `UCS Configuration is included in the Linux Command Reference PDF File` |
    | “How to copy file to a different location?”					| **Linux_Doc_Search**| `UCS Configuration is included in the Linux Command Reference PDF File` |
    | “What is the warranty period for a Cisco UCS C240 M5 blade?”	| **web_search** 	  | `search Web for ("Cisco UCS blade and Server warranty period")`         |
    | “What are the current prices for Cisco UCS B200 M5 servers?”  | **web_search**	  | `search Web for ("Cisco UCS blade and Server Pricing")`   				|
	| “Where can I find the Cisco UCS RMA form?”                    | **web_search** 	  | `search Web for ("Cisco UCS Service level agreement")`   				|
    | “What’s the latest stable release of Ubuntu?” 				| **web_search** 	  | `search Web for ("Linux flavors and releases specs and download repos")`|
    | “Where to download the latest release of Ubuntu?”				| **web_search** 	  | `search Web for ("Linux flavors and releases specs and download repos")`|

    ---

    **Your mission:**  
    - Accurately identify the user's intent.  
    - Choose the appropriate tool(s) following the strategy above.  
    - Deliver a clear, source‑backed answer that resolves the user's issue or request.  

    You may begin assisting the user now.
    """

    UCS_CLI_Config_Guide_Agent = LlmAgent(
        name="UCS_CLI_Config_Guide_Agent",
        model="gemini-2.5-flash",
        description="""Using the UCS CLI Configuration Guide PDF File, provide support for Cisco UCS System Configuration For Fabric Interconnect, UCS System Communication, UCS System DNS, UCS System LDAP and Authentication,
        Service Profile Configuration, Service Profile Template, vNIC Template, vHBA Template, UCS System Networking Port Channel and vlan, UCS Mac address Pool
        UCS System Storage Portchannel and vSAN, UCS Storage Zoning, UCS Fabric Interconnect Mode, UCS Fabric Interconnect Server Port, UCS Fabric Interconnect Uplink Port, 
        Manage UCS Licensing, UCS System Cooling and Fans, UCS System Power Supply and Power Managment, Blade Server configuration and IOM Port.""",
        instruction="""
		Provide Support for Cisco UCS System Configuration For Fabric Interconnect, UCS System Communication, UCS System DNS, UCS System LDAP and Authentication,
		Service Profile Configuration, Service Profile Template, vNIC Template, vHBA Template, UCS System Networking Port Channel and vlan, UCS Mac address Pool
		UCS System Storage Portchannel and vSAN, UCS Storage Zoning, UCS Fabric Interconnect Mode, UCS Fabric Interconnect Server Port, UCS Fabric Interconnect Uplink Port, 
		Manage UCS Licensing, UCS System Cooling and Fans, UCS System Power Supply and Power Managment, Blade Server configuration and IOM Port. 
		""",
        tools=[search_ucsm_cli_guide, add_document_to_rag]
    )
	
    Linux_CLI_Support_Agent = LlmAgent(
        name="Linux_CLI_Support_Agent",
        model="gemini-2.5-flash",
        description="""
		Provide Support for Linux Commands for Configuration, System and Network Administration, Boot Methods,
		Package Management, Bash Shell, Pattern Matching, Emacs Editor, sed editor, gawk programming, Source Code Management,
		Subversion Version Control, Virtualization CLI Tools.
		""",
        instruction="""
		Provide Support for Linux Commands for Configuration, System and Network Administration, Boot Methods,
		Package Management, Bash Shell, Pattern Matching, Emacs Editor, sed editor, gawk programming, Source Code Management,
		Subversion Version Control, Virtualization CLI Tools.
		""",
        tools=[search_linux_nutshell_guide, add_document_to_rag]
    )
	
    Web_Search_Agent = LlmAgent(
        name="Web_Search_Agent",
        model="gemini-2.5-flash",
        description="""
		This Web search is used when the question is vague and outside of the UCS CLI Configuration or Linux System Configuration Commands. 
		Perform Web Search for UCS Server Pricing, Server Warranty, Server End-of-life, Cisco UCS Support SLA, Release Notes and code bugs,
		Non-Cisco Server Brands. 
		Also, Use this agent to do websearch for Features of different Linux flavors, Stable release of each Linux Flavor, Release Note, Bug Report.
		""",
        instruction= """
		This Web search is used when the question is vague and outside of the UCS CLI Configuration or Linux System Configuration Commands. 
		Perform Web Search for UCS Server Pricing, Server Warranty, Server End-of-life, Cisco UCS Support SLA, Release Notes and code bugs,
		Non-Cisco Server Brands. 
		Also, Use this agent to do websearch for Features of different Linux flavors, Stable release of each Linux Flavor, Release Note, Bug Report.
		""",
        tools=[web_search]
    )

    # Create coordinator with sub-agents
    coordinatorAgent = LlmAgent(
        name="coordinator",
        model="gemini-2.5-flash",
        description="Main coordinator that delegates tasks to specialists.",
        instruction="""You are a coordinator. You must ALWAYS delegate tasks to specialists.

        IMPORTANT: Do not use tools yourself. Always use transfer_to_agent.

        Specialists:
        - UCS_CLI_Config_Guide_Agent: For ANY UCS CLI Configuration or Linux System Configuration Commands.
        - Linux_CLI_Support_Agent: For ANY information or research question
        - Web_Search_Agent: For ANY information or research question outside of the UCS CLI Configuration or Linux System Configuration Commands.

        Process:
        1. Analyze the user's request
        2. Decide which specialist to use
        3. IMMEDIATELY transfer using transfer_to_agent
        4. Do NOT attempt to answer yourself
        """,
        sub_agents=[UCS_CLI_Config_Guide_Agent, Linux_CLI_Support_Agent,Web_Search_Agent]
    )


    return coordinatorAgent


async def run_chatbot(coordinatorAgent, message: str, user_id: str = "customer1", session_id: str = "session1"):
    """
    Run the chatbot agent with conversation memory.

    Args:
        agent: The chatbot agent
        message: User message
        user_id: User identifier
        session_id: Session identifier for conversation memory
    """
    global global_session_service

    # Create or get session
    try:
        await global_session_service.create_session(
            app_name="customer_chatbot",
            user_id=user_id,
            session_id=session_id
        )
    except:
        pass  # Session already exists

    # Create runner
    runner = Runner(
        agent=coordinatorAgent,
        app_name="customer_chatbot",
        session_service=global_session_service
    )

    print(f"\n{'='*60}")
    print(f"User: {message}")
    print(f"{'='*60}\n")
    print("Agent: ", end="", flush=True)

    # Create message content
    message_content = types.Content(
        role="user",
        parts=[types.Part(text=message)]
    )

    # Run agent and collect response
    response_text = ""
    event_count = 0
    max_events = 25

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message_content
    ):
        event_count += 1

        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts') and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                        print(part.text, end='', flush=True)

        if event_count >= max_events:
            print("\n(Reached max events limit)")
            break

    print(f"\n{'='*60}\n")
    return response_text



async def run_background_builds():
    """Run RAG build tasks in the background."""
    print("[Background] Starting UCSM CLI RAG build...")
    await asyncio.to_thread(build_ucsm_cli_rag)
    
    # Load UCSM PDF (if present)
    if os.path.exists("UCSM_CLI_Configuration_Guide.pdf"):
        print("[Background] Loading UCSM_CLI_Configuration_Guide.pdf...")
        ucsm_docs = await asyncio.to_thread(load_documents, "UCSM_CLI_Configuration_Guide.pdf")
        ucsm_chunks = await asyncio.to_thread(text_splitter, ucsm_docs)
        for chunk in ucsm_chunks:
            await asyncio.to_thread(add_document_to_rag, chunk.page_content, chunk.metadata)

    # Load NetShell PDF (if present)
    if os.path.exists("Linux-in-a-Nutshell-6th-Edition.pdf"):
        print("[Background] Loading Linux-in-a-Nutshell-6th-Edition.pdf...")
        linux_docs = await asyncio.to_thread(load_documents, "Linux-in-a-Nutshell-6th-Edition.pdf")
        linux_chunks = await asyncio.to_thread(text_splitter, linux_docs)
        for chunk in linux_chunks:
            await asyncio.to_thread(add_document_to_rag, chunk.page_content, chunk.metadata)

    print("[Background] Starting Linux Nutshell RAG build...")
    await asyncio.to_thread(build_linux_nutshell_rag)
    
    print("✓ [Background] RAG Initialization Complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Google ADK Customer Chatbot Agent Startup")
    print("=" * 60)
    
    global coordinatorAgent
    
    # Initialize RAG system
    print("Initializing FAISS RAG system...")
    initialize_rag_system()

    # Start background tasks for heavy PDF processing
    asyncio.create_task(run_background_builds())

    # Create chatbot agent
    print("Creating multi_domain support engineer chatbot agent...")
    coordinatorAgent = create_multi_domain_support_engineer_chatbot_agent()
    print("✓ Agent created successfully")

    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "customer1"
    session_id: str = "session1"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not coordinatorAgent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Capture output from run_chatbot (which prints to stdout) 
    # run_chatbot returns the full response string at the end.
    response = await run_chatbot(coordinatorAgent, request.message, request.user_id, request.session_id)
    return {"response": response}

@app.get("/")
async def root():
    return {"status": "Agent is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for UI connection verification."""
    return {"status": "healthy", "active_sessions": 1}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # In Cloud Run, uvicorn should be run from the command line (CMD in Dockerfile).
    # This block is only for local testing.
    # We verify the module can be imported.
    try:
        from google.adk import Agent
        print(f"google.adk imported successfully")
    except ImportError as e:
        print(f"Error importing google.adk: {e}")
        
    uvicorn.run("chatbot_agent:app", host="0.0.0.0", port=port, reload=True)
