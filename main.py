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
        docs_df = pd.read_csv("internal_docs.csv")
        docs_df["last_updated"] = pd.Timestamp.now()  # Add a timestamp
        docs_df["tags"] = docs_df["question"].apply(
            lambda x: ["Legal"] if "legal" in x.lower() else ["Compliance"] if "compliance" in x.lower() else []
        )
        return docs_df
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
    relevant_docs = internal_docs_df.iloc[top_k_indices].copy()
    relevant_docs["confidence"] = similarities[top_k_indices]  # Add confidence scores
    return relevant_docs

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

# Tab 2: Escalation Center
with tab2:
    st.markdown("### 🚨 Escalation Management")
    
    # Transparency Panel
    st.markdown("#### 📊 AI Governance & Transparency")
    if not internal_docs_df.empty:
        last_updated = internal_docs_df["last_updated"].max()
    else:
        last_updated = "N/A"
    
    st.markdown(f"""
        <div class='metric-card'>
            <p><strong>Last Knowledge Update:</strong> {last_updated}</p>
            <p><strong>Escalations This Week:</strong> {len(st.session_state.escalations)}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create New Escalation
    with st.expander("Create New Escalation", expanded=True):
        esc_col1, esc_col2 = st.columns([2, 1])
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
    st.markdown("#### Active Escalations")
    if st.session_state.escalations:
        for idx, esc in enumerate(st.session_state.escalations):
            with st.container():
                st.markdown(f"""
                    <div class='metric-card'>
                        <h4>{esc['reason']} Escalation - {esc['priority']}</h4>
                        <p><strong>Query:</strong> {esc['query']}</p>
                        <p style='color: #ea580c;'>Status: {esc['status']}</p>
                        <button onclick="markResolved({idx})">Mark Resolved</button>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active escalations")
    
    # Escalation Trends Visualization
    st.markdown("#### Escalation Trends")
    escalation_df = pd.DataFrame(st.session_state.escalations)
    if not escalation_df.empty:
        st.bar_chart(escalation_df["reason"].value_counts())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Built by Ankita Avadhani using Claude 3.5 Sonnet, Streamlit, and RAG
    </div>
""", unsafe_allow_html=True)
