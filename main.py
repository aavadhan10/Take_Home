
Copy
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic

# Move page config to the top
st.set_page_config(page_title="Moxie AI Support Agent", page_icon="🚀", layout="wide")

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["anthropic_api_key"]
except Exception as e:
    st.error(f"Error loading API key: {e}")
    api_key = None

# Initialize Anthropic client
try:
    client = Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing Anthropic client: {e}")
    client = None

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
    try:
        return pd.read_csv("internal_docs.csv")
    except FileNotFoundError:
        st.error("Error: 'internal_docs.csv' not found. Please ensure the file exists.")
        return pd.DataFrame()

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
    if client is None:
        st.error("Anthropic client not initialized. Unable to generate response.")
        return "Error: AI assistant is currently unavailable.", pd.DataFrame()

    try:
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
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text, relevant_docs
    
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return f"Error: Unable to generate response. Details: {str(e)}", relevant_docs

# Escalation logic
def determine_escalation(query):
    compliance_keywords = ["legal", "compliance", "regulation", "privacy", "confidentiality", "lawsuit", "liability"]
    if any(keyword in query.lower() for keyword in compliance_keywords):
        return True, "Compliance Review Needed"
    return False, "Standard Query"

# Initialize session state
if 'queries_handled' not in st.session_state:
    st.session_state.queries_handled = 0
if 'queries_escalated' not in st.session_state:
    st.session_state.queries_escalated = 0

# Title and Overview
st.title("🚀 Moxie AI Support Agent")
st.markdown("### Empowering Provider Success Managers with AI-powered assistance")

# Sidebar for User Interactions and Metrics
with st.sidebar:
    st.header("🤖 AI Agent Dashboard")
    
    # Communication channel selection
    st.subheader("📞 Communication Channel")
    support_channel = st.radio(
        "Select Channel",
        ["Chat Support", "Email Response", "SMS Handling", "Help Center Ticket"],
        key="support_channel"
    )
    st.write(f"Selected Channel: **{support_channel}**")

    # Provider information lookup
    st.subheader("🔍 Find Provider Information")
    provider_data = {
        "Provider 1": {"Email": "provider1@example.com", "Phone": "123-456-7890"},
        "Provider 2": {"Email": "provider2@example.com", "Phone": "987-654-3210"},
    }
    provider_name = st.selectbox("Select Provider", list(provider_data.keys()))
    if provider_name:
        st.write(f"**Email:** {provider_data[provider_name]['Email']}")
        st.write(f"**Phone:** {provider_data[provider_name]['Phone']}")

    # Performance metrics
    st.subheader("📊 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Answered", st.session_state.queries_handled)
    with col2:
        st.metric("Questions Escalated", st.session_state.queries_escalated)

# Main Interface
st.header("🔍 What can we help you with today?")
psm_query = st.text_input("Ask your question", placeholder="Type your question here...")

# Example query buttons
st.markdown("**Need inspiration? Try these examples:**")
example_cols = st.columns(3)
example_queries = [
    "How do I update billing info?",
    "What are the marketing guidelines?",
    "How do I handle patient data?"
]
for col, query in zip(example_cols, example_queries):
    if col.button(query):
        psm_query = query

# Query Processing
if psm_query:
    if api_key is None or client is None:
        st.error("AI assistant is not configured. Please check your API key.")
    else:
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
        st.markdown("### 🤖 Here’s what I found:")
        st.info(response)
        
        # Escalation Buttons
        with st.expander("🚨 Escalation Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Escalate to Compliance"):
                    st.session_state.escalation_status = "Compliance"
                    st.success("This has been escalated to the Compliance team!")
            with col2:
                if st.button("Escalate to Legal"):
                    st.session_state.escalation_status = "Legal"
                    st.success("This has been escalated to the Legal team!")
            with col3:
                if st.button("Escalate to Finance"):
                    st.session_state.escalation_status = "Finance"
                    st.success("This has been escalated to the Finance team!")

        # Retrieved Documents
        with st.expander("📚 Relevant Documentation"):
            for idx, row in relevant_docs.iterrows():
                st.markdown(f"**{row['question']}**")
                st.write(row['answer'])
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using **Claude 3.5 Sonnet**, **Streamlit**, and **RAG**")
