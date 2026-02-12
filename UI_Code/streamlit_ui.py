#streamlit_ui.py

"""
Streamlit UI for System Support Engineer Chatbot Agent

Tool for assisting system support engineers with an intuitive interface for chatting with the chatbot.
Can connect to local FastAPI or Cloud Run deployment.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional
import os

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="System Support Engineer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Styling
# ============================================================================

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196F3;
        color: #000000 !important;
    }
    .assistant-message {
        background-color: #ffffff !important;
        border-left: 4px solid #4CAF50;
        color: #000000 !important;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #000000 !important;
    }
    .info-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .rag-badge {
        background-color: #c8e6c9;
        color: #1b5e20;
    }
    .web-badge {
        background-color: #bbdefb;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_session_{datetime.now().timestamp()}"

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8080"

if "connected" not in st.session_state:
    st.session_state.connected = False

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    # API Configuration
    st.subheader("🌐 API Connection")

    api_option = st.radio(
        "Select API location:",
        ["Local (localhost)", "Cloud Run", "Custom URL"]
    )

    if api_option == "Local (localhost)":
        col_local, col_port = st.columns([3, 1])
        with col_local:
            local_host = st.text_input("Host", value="http://localhost", disabled=True)
        with col_port:
            local_port = st.text_input("Port", value="8080")
        
        st.session_state.api_url = f"{local_host}:{local_port}"
        st.info(f"Using local API at {st.session_state.api_url}")

    elif api_option == "Cloud Run":
        cloud_url = st.text_input(
            "Cloud Run URL",
            placeholder="https://your-project.run.app",
            help="Enter your Cloud Run deployment URL"
        )
        if cloud_url:
            st.session_state.api_url = cloud_url.rstrip("/")

    else:
        custom_url = st.text_input(
            "Custom API URL",
            value=st.session_state.api_url,
            help="Enter custom API URL"
        )
        if custom_url:
            st.session_state.api_url = custom_url.rstrip("/")

    # Test Connection
    if st.button("🔌 Test Connection", use_container_width=True):
        try:
            # Try health endpoint
            try_urls = [f"{st.session_state.api_url}/health", f"{st.session_state.api_url}/"]
            success = False
            last_error = None
            
            for url in try_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        st.session_state.connected = True
                        data = response.json()
                        st.success("✓ Connected!")
                        if "status" in data:
                            st.write(f"Status: {data['status']}")
                        if "active_sessions" in data:
                            st.write(f"Active sessions: {data['active_sessions']}")
                        success = True
                        break
                    else:
                        last_error = f"Status {response.status_code}"
                except Exception as e:
                    last_error = str(e)
            
            if not success:
                st.session_state.connected = False
                st.error(f"✗ Connection failed: {last_error}")
                if "404" in str(last_error):
                    st.warning("Hint: Check if the URL is correct and the service is deployed.")
                    
        except Exception as e:
            st.session_state.connected = False
            st.error(f"✗ Connection failed: {str(e)}")

    st.divider()

    # Session Information
    st.subheader("📋 Session Info")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Connected:** {'✓ Yes' if st.session_state.connected else '✗ No'}")

    st.divider()

    # Knowledge Base Management
    st.subheader("📚 Knowledge Base")

    with st.expander("Add Document"):
        doc_content = st.text_area(
            "Document content:",
            height=100,
            placeholder="Enter document content..."
        )
        doc_category = st.text_input(
            "Category:",
            placeholder="e.g., policy, shipping, support"
        )

        if st.button("➕ Add Document", use_container_width=True):
            if not doc_content:
                st.error("Please enter document content")
            elif not st.session_state.connected:
                st.error("API not connected")
            else:
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/add-document",
                        json={
                            "content": doc_content,
                            "metadata": {"category": doc_category}
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✓ {data['message']}")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to add document: {str(e)}")

    with st.expander("Search Knowledge Base"):
        search_query = st.text_input(
            "Search query:",
            placeholder="What do you want to search for?"
        )

        if st.button("🔍 Search", use_container_width=True):
            if not search_query:
                st.error("Please enter a search query")
            elif not st.session_state.connected:
                st.error("API not connected")
            else:
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/search",
                        json={"query": search_query, "top_k": 3},
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.write(f"**Found {len(data['results'])} results:**")
                        for i, result in enumerate(data['results'], 1):
                            st.write(f"{i}. {result['content'][:100]}...")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared")
        st.rerun()

# ============================================================================
# Main Content
# ============================================================================

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🤖 System Support Engineer Chatbot")
    st.caption("System Support Engineer Chatbot with RAG and web search To help system support engineers quickly find answers using company knowledge base and real-time web search.")

with col2:
    if st.session_state.connected:
        st.success("🟢 Connected")
    else:
        st.warning("🔴 Not Connected")

st.divider()

# Chat Messages Display
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            tool_badge = ""
            if "tool_used" in message and message["tool_used"]:
                tool_class = "rag-badge" if message["tool_used"] == "RAG" else "web-badge"
                tool_badge = f'<span class="info-badge {tool_class}">{message["tool_used"]}</span>'

            st.markdown(
                f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {tool_badge}
                    <br>{message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

# Input Area
st.divider()
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Message:",
        placeholder="Ask me anything about our services...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True)

# Handle Message Sending
if send_button and user_input:
    if not st.session_state.connected:
        st.error("❌ API not connected. Please configure connection in sidebar.")
    else:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Get response from API
        try:
            with st.spinner("🔄 Getting response..."):
                response = requests.post(
                    f"{st.session_state.api_url}/chat",
                    json={
                        "user_id": "streamlit_user",
                        "session_id": st.session_state.session_id,
                        "message": user_input
                    },
                    timeout=30
                )

            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["response"],
                    "tool_used": data.get("tool_used")
                })
                st.rerun()
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", f"Status {response.status_code}")
                st.error(f"❌ Error: {error_msg}")

        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error. Please check the API URL.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# Info Section
# ============================================================================

if len(st.session_state.messages) == 0:
    st.info("""
    ### Welcome to Customer Chatbot! 👋

    This chatbot uses **Google ADK with Gemini 2.5 Flash** to provide intelligent customer support.

    **Features:**
    - 🧠 **Intelligent Tool Selection** - Automatically chooses between Documentation and web search
    - 📚 **RAG System** - Semantic search over company knowledge base
    - 🌐 **Web Search** - Real-time external information
    - 💬 **Conversation Memory** - Remembers previous messages
    - 🔌 **REST API** - Easy integration with any application

    **How to Use:**
    1. Configure your API connection in the sidebar
    2. Test the connection
    3. Start asking questions!

    **Example Questions:**
    - "What are the Steps to configure Primary Fabric Interconnect"	
    - "What are the steps to create LAN Port Channel?"              
    - "What are the steps to create Service Profile Template?"      
    - "What are the steps to create vNIC Template?"                 
    - "How to list files and subfolders in current Directory?"      
    - "How to copy file to a different location?"					
    - "How to move file to a different location?"					
    - "What is the warranty period for a Cisco UCS C240 M5 blade?"	
    - "What are the current prices for Cisco UCS B200 M5 servers?"  
    - "Where can I find the Cisco UCS RMA form?"                    
    - "What’s the latest stable release of Ubuntu?" 				
    - "Where to download the latest release of Ubuntu?"				
    """)

# ============================================================================
# Footer
# ============================================================================

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📖 [Documentation](https://github.com/your-repo)")

with col2:
    st.caption("🐛 [Report Issues](https://github.com/your-repo/issues)")

with col3:
    st.caption("💬 Chat Sessions: " + str(len(st.session_state.messages) // 2))
