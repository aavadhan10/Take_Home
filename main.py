import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic

# Page Configuration
st.set_page_config(
    page_title="Moxie AI Support Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* General Styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 12px 20px;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    
    /* Tab Styling */
    .stTabs > div > div > div {
        gap: 8px;
        padding: 10px 0;
    }
    
    /* Response Container */
    .response-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 20px 0;
    }
    
    /* Example Query Buttons */
    .example-query {
        background-color: #f1f5f9;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        display: inline-block;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .example-query:hover {
        background-color: #e2e8f0;
    }
    
    /* Channel Selection */
    .channel-select {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
    }
    .channel-select:hover {
        background-color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load API key and initialize Anthropic client
try:
    api_key = st.secrets["anthropic_api_key"]
    client = Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing: {e}")
    api_key = None
    client = None

# Embedding and model functions
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return tokenizer, model

tokenizer, model = load_embedding_model()

# Load and prepare documents
@st.cache_data
def load_docs():
    try:
        return pd.read_csv("internal_docs.csv")
    except FileNotFoundError:
        st.error("Error: 'internal_docs.csv' not found.")
        return pd.DataFrame()

def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()

internal_docs_df = load_docs()
doc_embeddings = get_embeddings(internal_docs_df["question"].tolist()) if not internal_docs_df.empty else None

def retrieve_documents(query, top_k=3):
    query_embedding = get_embeddings([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return internal_docs_df.iloc[top_k_indices]

# Enhanced RAG with Claude
def ask_claude_with_rag(query):
    if client is None:
        return "Error: AI assistant is unavailable.", pd.DataFrame()
    
    try:
        with st.spinner("🔍 Searching through documentation..."):
            relevant_docs = retrieve_documents(query)
            context = "\n".join(relevant_docs["question"] + ": " + relevant_docs["answer"])
            
            prompt = f"""
            You are an AI assistant for Moxie, supporting Provider Success Managers (PSMs) and medical spa providers.

            Context from internal documentation:
            {context}

            Provide a helpful, professional response to: {query}

            If the query involves compliance, legal, or specialized expertise, indicate escalation needs.
            """
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text, relevant_docs
    except Exception as e:
        return f"Error: {str(e)}", relevant_docs

# Initialize session state
if 'queries_handled' not in st.session_state:
    st.session_state.queries_handled = 0
if 'queries_escalated' not in st.session_state:
    st.session_state.queries_escalated = 0
if 'escalations' not in st.session_state:
    st.session_state.escalations = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #0f172a;'>🤖 AI Support</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics
    metrics_cols = st.columns(2)
    with metrics_cols[0]:
        st.markdown("""
            <div class='metric-card'>
                <p style='color: #64748b; margin: 0;'>Questions Answered</p>
                <h2 style='color: #0284c7; margin: 0;'>{}</h2>
            </div>
        """.format(st.session_state.queries_handled), unsafe_allow_html=True)
    
    with metrics_cols[1]:
        st.markdown("""
            <div class='metric-card'>
                <p style='color: #64748b; margin: 0;'>Escalated</p>
                <h2 style='color: #ea580c; margin: 0;'>{}</h2>
            </div>
        """.format(st.session_state.queries_escalated), unsafe_allow_html=True)
    
    # Provider Contact Section
    st.markdown("### 📱 Contact Provider")
    
    # Sample provider data
    provider_data = {
        "Provider 1: Jesse Lau": {"email": "provider1@moxie.com", "phone": "123-456-7890", "preferred": "email"},
        "Provider 2: Dan Friedman": {"email": "provider2@moxie.com", "phone": "987-654-3210", "preferred": "sms"},
        "Provider 3 Kamau Massey": {"email": "provider3@moxie.com", "phone": "555-123-4567", "preferred": "chat"}
    }
    
    selected_provider = st.selectbox("Select Provider", list(provider_data.keys()))
    
    if selected_provider:
        st.markdown("""
            <div class='metric-card'>
                <p><strong>📧 Email:</strong> {}</p>
                <p><strong>📱 Phone:</strong> {}</p>
                <p><strong>⭐ Preferred Channel:</strong> {}</p>
            </div>
        """.format(
            provider_data[selected_provider]["email"],
            provider_data[selected_provider]["phone"],
            provider_data[selected_provider]["preferred"].upper()
        ), unsafe_allow_html=True)

        st.markdown("### 📤 Send Message")
        
        selected_channel = st.radio(
            "Select Communication Channel:",
            ["💬 Chat Support", "📧 Email", "📱 SMS", "❓ Help Center"],
            key="channel_select",
        )

        # Message composition
        message = st.text_area("Message:", placeholder="Type your message here...", height=100)
        
        # Channel-specific inputs
        if selected_channel == "💬 Chat Support":
            if st.button("Start Chat Session", type="primary"):
                st.success(f"Opening chat session with {selected_provider}...")
                
        elif selected_channel == "📧 Email":
            subject = st.text_input("Subject:", placeholder="Enter email subject")
            if st.button("Send Email", type="primary"):
                st.success(f"Email sent to {provider_data[selected_provider]['email']}")
                
        elif selected_channel == "📱 SMS":
            if st.button("Send SMS", type="primary"):
                st.success(f"SMS sent to {provider_data[selected_provider]['phone']}")
                
        elif selected_channel == "❓ Help Center":
            ticket_priority = st.select_slider(
                "Ticket Priority",
                options=["Low", "Medium", "High", "Urgent"]
            )
            if st.button("Create Help Center Ticket", type="primary"):
                st.success(f"Help Center ticket created for {selected_provider}")

        # Send Message Button
        if message and st.button("Send Message", type="primary"):
            st.session_state.message_history.append({
                "provider": selected_provider,
                "channel": selected_channel,
                "message": message,
                "timestamp": pd.Timestamp.now()
            })
            st.success(f"Message sent to {selected_provider} via {selected_channel}")

# Main Content Area
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🚀 Moxie AI Support Agent</h1>
        <p style='color: #64748b;'>Empowering Provider Success Managers with AI assistance (Powered by Claude 3.5 Sonnet & RAG Technology) </p>
    </div>
""", unsafe_allow_html=True)

# Create tabs with enhanced styling
tab1, tab2, tab3 = st.tabs([
    "🔍 AI Support Question Assistant",
    "🚨 Escalation Center",
    "📊 Documentation Search + Interaction Insights"
])

# Tab 1: AI Support Question Assistant
with tab1:
    # Search Section
    st.markdown("### How can we help you today?")
    query_col1, query_col2 = st.columns([4,1])
    with query_col1:
        psm_query = st.text_input("", placeholder="Type your question here...", key="main_search")
    with query_col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Example Queries
    st.markdown("##### Quick Access Questions")
    example_queries = [
        "How do I update billing info?",
        "What are the marketing guidelines?",
        "How do I handle patient data?",
        "Reset password",
        "Business hours",
        "Access dashboard"
    ]
    
    example_cols = st.columns(3)
    for i, query in enumerate(example_queries):
        with example_cols[i % 3]:
            if st.button(f"💡 {query}", key=f"example_{i}"):
                psm_query = query

    # Process Query and Display Response
    if psm_query:
        response, relevant_docs = ask_claude_with_rag(psm_query)
        
        # Display Response
        st.markdown("""
            <div class='response-container'>
                <h4>🤖 AI Assistant Response</h4>
                <p>{}</p>
            </div>
        """.format(response), unsafe_allow_html=True)
        
        # Related Documentation
        with st.expander("📚 Relevant Internal Documentation"):
            st.dataframe(
                relevant_docs[["question", "answer"]],
                use_container_width=True,
                column_config={
                    "question": "Question",
                    "answer": "Answer"
                }
            )
        
        # Update metrics
        st.session_state.queries_handled += 1
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": psm_query,
            "response": response,
            "channel": selected_channel
        })
    # Tab 2: Escalation Center
with tab2:
    st.markdown("### 🚨 Escalation Management")
    
    # Create New Escalation
    with st.expander("Create New Escalation", expanded=True):
        esc_col1, esc_col2 = st.columns([2,1])
        with esc_col1:
            escalation_query = st.text_input("Query to Escalate")
            escalation_reason = st.selectbox(
                "Reason",
                ["Compliance", "Legal", "Technical", "Other"]
            )
        with esc_col2:
            priority = st.select_slider(
                "Priority",
                ["Low", "Medium", "High", "Urgent"]
            )
            if st.button("🚨 Create Escalation", type="primary"):
                st.session_state.escalations.append({
                    "query": escalation_query,
                    "reason": escalation_reason,
                    "priority": priority,
                    "status": "Pending"
                })
                st.session_state.queries_escalated += 1
                st.success("Escalation created!")
    
    # View Escalations
    if st.session_state.escalations:
        for idx, esc in enumerate(st.session_state.escalations):
            with st.container():
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4>{esc['reason']} Escalation - {esc['priority']}</h4>
                        <p>{esc['query']}</p>
                        <p style='color: #ea580c;'>Status: {esc['status']}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active escalations")

# Tab 3: Common Documentation + Interaction Insights
with tab3:
    st.markdown("### 📊 Knowledge Base & Interactions")
    
    # Relevant Documents Section
    st.subheader("📚 Internal Documentation")
    if not internal_docs_df.empty:
        doc_search = st.text_input("Search documentation...", key="doc_search")
        if doc_search:
            filtered_docs = internal_docs_df[
                internal_docs_df["question"].str.contains(doc_search, case=False) |
                internal_docs_df["answer"].str.contains(doc_search, case=False)
            ]
        else:
            filtered_docs = internal_docs_df
        
        st.dataframe(
            filtered_docs,
            use_container_width=True,
            column_config={
                "question": "Topic/Question",
                "answer": "Information/Answer"
            }
        )
    
    # Recent Message History
    st.subheader("Recent Messages")
    if st.session_state.message_history:
        for msg in reversed(st.session_state.message_history[-5:]):
            st.markdown(f"""
                <div class='metric-card'>
                    <p><strong>To:</strong> {msg['provider']}</p>
                    <p><strong>Channel:</strong> {msg['channel']}</p>
                    <p><strong>Message:</strong> {msg['message']}</p>
                    <p><small>Sent: {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}</small></p>
                </div>
            """, unsafe_allow_html=True)
    
    # Chat History
    st.subheader("Recent AI Interactions")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history[-5:]:  # Show last 5 interactions
            st.markdown(f"""
                <div class='metric-card'>
                    <p><strong>Question:</strong> {chat['query']}</p>
                    <p><strong>Response:</strong> {chat['response'][:200]}...</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Escalation Analytics
    if st.session_state.escalations:
        st.subheader("Escalation Analytics")
        escalation_df = pd.DataFrame(st.session_state.escalations)
        st.dataframe(
            escalation_df,
            use_container_width=True,
            column_config={
                "query": "Query",
                "reason": "Reason",
                "priority": "Priority",
                "status": "Status"
            }
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Built with ❤️ using Claude 3.5 Sonnet, Streamlit, and RAG
    </div>
""", unsafe_allow_html=True)
