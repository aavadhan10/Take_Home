import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic
import os
import json

# Page configuration
st.set_page_config(page_title="Moxie AI Support Agent", page_icon="🚀", layout="wide")

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["anthropic_api_key"]
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
    return pd.read_csv("internal_docs.csv")

@st.cache_data
def load_provider_queries():
    return pd.read_csv("provider_queries.csv")

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
        st.error("AI client not initialized")
        return "Error: AI assistant unavailable", pd.DataFrame()

    try:
        relevant_docs = retrieve_documents(query)
        context = "\n".join(relevant_docs["question"] + ": " + relevant_docs["answer"])
        
        full_prompt = f"""
        You are an AI assistant for Moxie, supporting Provider Success Managers.
        Context from internal documentation:
        {context}
        Provide a helpful response to: {query}
        """
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text, relevant_docs
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Error: {str(e)}", pd.DataFrame()

# Initialize session state
if 'queries_handled' not in st.session_state:
    st.session_state.queries_handled = 0
if 'queries_escalated' not in st.session_state:
    st.session_state.queries_escalated = 0
if 'escalations' not in st.session_state:
    st.session_state.escalations = []

# Escalation logic
def determine_escalation(query):
    compliance_keywords = [
        "legal", "compliance", "regulation", "privacy", 
        "confidentiality", "lawsuit", "liability"
    ]
    
    complexity_keywords = [
        "complex", "strategy", "advanced", "comprehensive", 
        "detailed analysis", "extensive"
    ]
    
    if any(keyword in query.lower() for keyword in compliance_keywords):
        return True, "Compliance Review Needed"
    
    if any(keyword in query.lower() for keyword in complexity_keywords):
        return True, "Expert Review Required"
    
    return False, "Standard Query"

# Main Application
def main():
    # Title and Introduction
    st.title("🚀 Moxie AI Support Agent")
    st.markdown("""
        ### Empowering Provider Success Managers
        
        Reduce workload, handle queries efficiently, and focus on critical business challenges.
    """)

    # Sidebar Navigation
    with st.sidebar:
        st.header("🤖 AI Agent Toolkit")
        feature = st.radio("Choose Interaction Mode", [
            "Query Assistance",
            "Escalation Center", 
            "Communication Channels",
            "Query Library",
            "Performance Insights"
        ])

    # Feature-specific implementations
    if feature == "Query Assistance":
        st.header("🔍 Provider Query Assistance")
        
        # Query Input
        psm_query = st.text_input("Enter a provider query", 
            placeholder="e.g., How do I update billing information?"
        )
        
        # Example Quick Queries
        st.markdown("**Quick Query Examples:**")
        example_cols = st.columns(3)
        example_queries = [
            "Billing update process",
            "Marketing compliance",
            "Dashboard access"
        ]
        for col, query in zip(example_cols, example_queries):
            if col.button(query):
                psm_query = query
        
        # AI-Powered Response
        if psm_query:
            # Determine escalation
            needs_escalation, escalation_reason = determine_escalation(psm_query)
            
            # Generate AI Response
            response, relevant_docs = ask_claude_with_rag(psm_query)
            
            # Display Response
            st.markdown("### 🤖 AI Agent Response")
            st.info(response)
            
            # Escalation Handling
            if needs_escalation:
                st.warning(f"🚨 {escalation_reason}")
                st.session_state.queries_escalated += 1
            else:
                st.session_state.queries_handled += 1
            
            # Retrieved Documents
            with st.expander("📚 Relevant Documentation"):
                st.table(relevant_docs)

    elif feature == "Escalation Center":
        st.header("🚨 Escalation Management")
        
        # Escalation Type Selection
        escalation_types = [
            "Legal Compliance",
            "Financial Review",
            "Marketing Support",
            "Technical Issues",
            "Business Coaching",
            "Patient Data Privacy"
        ]
        
        selected_type = st.selectbox(
            "Select Escalation Category", 
            escalation_types
        )
        
        # Escalation Details
        escalation_details = st.text_area(
            "Provide Detailed Context for Escalation",
            height=200
        )
        
        # Create Escalation Ticket
        if st.button("Create Escalation Ticket"):
            ticket_id = f"MOXIE-{np.random.randint(1000, 9999)}"
            
            escalation_record = {
                'ticket_id': ticket_id,
                'type': selected_type,
                'details': escalation_details
            }
            
            st.session_state.escalations.append(escalation_record)
            st.success(f"Escalation Ticket Created: {ticket_id}")

    elif feature == "Communication Channels":
        st.header("📡 Provider Communication Channels")
        
        # Channel Selection
        channel = st.radio("Select Communication Method", [
            "Chat Support",
            "Email Response",
            "SMS Handling",
            "Help Center Ticket"
        ])
        
        # Channel-Specific Inputs
        if channel == "Chat Support":
            st.write("🤖 Chat Support Simulation")
            chat_query = st.text_input("Enter Provider Query")
            if chat_query:
                st.info("AI-Generated Chat Response Placeholder")
        
        elif channel == "Email Response":
            st.write("📧 Email Response Generator")
            email_context = st.text_area("Provide Email Context")
            if st.button("Generate Email Draft"):
                st.code("AI-Generated Email Draft Placeholder")

    elif feature == "Query Library":
        st.header("🗂️ Provider Query Reference")
        
        query_categories = {
            "Billing": [
                "Update payment method",
                "Understand billing cycles"
            ],
            "Technical Support": [
                "Software integration",
                "Dashboard access"
            ],
            "Compliance": [
                "HIPAA regulations",
                "Marketing guidelines"
            ]
        }
        
        for category, queries in query_categories.items():
            with st.expander(category):
                for query in queries:
                    st.write(f"- {query}")

    else:  # Performance Insights
        st.header("📊 PSM Efficiency Metrics")
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Queries Handled", st.session_state.queries_handled)
        
        with col2:
            st.metric("Queries Escalated", st.session_state.queries_escalated)
        
        with col3:
            escalation_rate = (
                st.session_state.queries_escalated / 
                (st.session_state.queries_handled + 1) * 100
            )
            st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
        
        # Recent Escalations
        st.subheader("Recent Escalation Tickets")
        if st.session_state.escalations:
            escalation_df = pd.DataFrame(st.session_state.escalations)
            st.dataframe(escalation_df)
        else:
            st.write("No recent escalations")

    # Footer
    st.markdown("---")
    st.markdown("**Moxie AI Support Agent** - Empowering Provider Success Managers")

# Run the application
if __name__ == "__main__":
    main()
