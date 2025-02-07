import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic
import os
import json

# Move page config to the top
st.set_page_config(page_title="Moxie AI Support Agent", page_icon="🚀", layout="wide")

# Remove the main() function wrapper
# Load API key from Streamlit secrets
api_key = st.secrets["anthropic_api_key"]
os.environ["ANTHROPIC_API_KEY"] = api_key

# Initialize Anthropic client
client = Anthropic()

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load internal documentation
@st.cache_data
def load_docs():
    return pd.read_csv("internal_docs.csv")

@st.cache_data
def load_provider_queries():
    return pd.read_csv("provider_queries.csv")

# Load predefined query types
@st.cache_data
def load_query_types():
    return {
        "Routine": [
            "How do I update my billing information?",
            "What are the business hours for support?",
            "How do I access my dashboard?"
        ],
        "Compliance": [
            "Are there any legal restrictions on marketing?",
            "What are the data privacy guidelines?",
            "How do I handle patient confidentiality?"
        ],
        "Complex": [
            "I'm experiencing issues with patient management software",
            "How can I optimize my medspa's marketing strategy?",
            "What financial reporting do I need to maintain?"
        ]
    }

# Embedding and retrieval functions
def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()

# Load and prepare embeddings
internal_docs_df = load_docs()
doc_embeddings = get_embeddings(internal_docs_df["question"].tolist())

# Retrieve documents
def retrieve_documents(query, top_k=3):
    query_embedding = get_embeddings([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return internal_docs_df.iloc[top_k_indices]

# RAG with Claude
def ask_claude_with_rag(query):
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs["question"] + ": " + relevant_docs["answer"])
    
    full_prompt = f"""
    You are an AI assistant for Moxie, supporting Provider Success Managers (PSMs) and medical spa providers.

    Context from internal documentation:
    {context}

    Provide a helpful, professional response to the following query:
    {query}

    If the query involves sensitive topics like compliance, legal, or requires specialized expertise, indicate it needs escalation.
    """
    
    response = client.messages.create(
        model="claude-3.5-sonnet",
        max_tokens=500,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    return response.content[0].text, relevant_docs

# Escalation logic
def determine_escalation(query):
    compliance_keywords = [
        "legal", "compliance", "regulation", "privacy", 
        "confidentiality", "lawsuit", "liability"
    ]
    
    if any(keyword in query.lower() for keyword in compliance_keywords):
        return True, "Compliance Review Needed"
    
    complexity_keywords = [
        "complex", "strategy", "advanced", "comprehensive", 
        "detailed analysis", "extensive"
    ]
    
    if any(keyword in query.lower() for keyword in complexity_keywords):
        return True, "Expert Review Required"
    
    return False, "Standard Query"

# Initialize session state
if 'queries_handled' not in st.session_state:
    st.session_state.queries_handled = 0
if 'queries_escalated' not in st.session_state:
    st.session_state.queries_escalated = 0

# Title and Overview
st.title("🚀 Moxie AI Support Agent")
st.markdown("### Empowering Provider Success Managers")

# Sidebar for User Interactions and Metrics
with st.sidebar:
    st.header("🤖 AI Agent Dashboard")
    
    # PSM-Facing Metrics
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries Handled", st.session_state.queries_handled)
    with col2:
        st.metric("Queries Escalated", st.session_state.queries_escalated)
    
    # Example Query Types
    st.subheader("Query Type Examples")
    query_types = load_query_types()
    for category, queries in query_types.items():
        with st.expander(f"{category} Queries"):
            for q in queries:
                st.write(f"- {q}")
    
    # Feedback Mechanism
    st.subheader("Your Feedback")
    feedback = st.radio("How is the AI agent helping?", 
                        ["👍 Very Helpful", "👀 Needs Improvement", "🤔 Neutral"])
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Query Interface")
    
    # Query Input with Examples
    query_placeholder = "Ask a question about your medical spa business..."
    psm_query = st.text_input("Enter Your Query", placeholder=query_placeholder)
    
    # Example Query Buttons
    st.markdown("**Quick Examples:**")
    example_cols = st.columns(3)
    example_queries = [
        "How do I update billing info?",
        "Marketing compliance guidelines",
        "Patient data privacy"
    ]
    for col, query in zip(example_cols, example_queries):
        if col.button(query):
            psm_query = query

with col2:
    st.header("📋 Query Details")
    # Placeholder for query details
    query_details_container = st.container()

# Query Processing
if psm_query:
    # Determine if escalation is needed
    needs_escalation, escalation_reason = determine_escalation(psm_query)
    
    # Generate AI Response
    response, relevant_docs = ask_claude_with_rag(psm_query)
    
    # Update Metrics
    if needs_escalation:
        st.session_state.queries_escalated += 1
    else:
        st.session_state.queries_handled += 1
    
    # Display Response
    st.markdown("### 🤖 AI Agent Response")
    st.info(response)
    
    # Query Details
    with query_details_container:
        st.markdown("**Query Analysis**")
        st.write(f"**Type:** {'Escalated' if needs_escalation else 'Handled'}")
        st.write(f"**Reason:** {escalation_reason}")
    
    # Retrieved Documents
    with st.expander("📚 Relevant Documentation"):
        st.table(relevant_docs)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using **Claude 3.5 Sonnet**, **Streamlit**, and **RAG**")
