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

# Predefined provider database (simulated)
PROVIDER_DATABASE = [
    {
        "Name": "Dr. Jane Smith",
        "Medical Spa": "Wellness Medspa",
        "Location": "Los Angeles, CA",
        "Contact": {
            "Email": "jane@wellnessmedspa.com",
            "Phone": "(555) 123-4567",
            "SMS": "(555) 123-4567"
        },
        "Account Number": "MOXIE-2023"
    },
    {
        "Name": "Dr. Mike Johnson",
        "Medical Spa": "Precision Aesthetics",
        "Location": "Chicago, IL",
        "Contact": {
            "Email": "mike@precisionaesthetics.com",
            "Phone": "(555) 987-6543",
            "SMS": "(555) 987-6543"
        },
        "Account Number": "MOXIE-2024"
    }
]

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

# Provider Information Lookup
def provider_information_lookup():
    st.subheader("🔍 Provider Information Finder")
    
    # Search inputs
    search_type = st.selectbox("Search By", [
        "Name", 
        "Medical Spa", 
        "Location", 
        "Account Number"
    ])
    
    search_query = st.text_input(f"Enter {search_type}")
    
    if search_query:
        # Filter providers
        results = [
            provider for provider in PROVIDER_DATABASE 
            if search_query.lower() in str(provider).lower()
        ]
        
        if results:
            for provider in results:
                with st.expander(f"{provider['Name']} - {provider['Medical Spa']}"):
                    st.json(provider)
        else:
            st.warning("No providers found matching your search.")

# Communication Channels
def communication_channels():
    st.header("📡 Provider Communication")
    
    # Provider Lookup
    provider_information_lookup()
    
    # Communication Method Selection
    communication_method = st.selectbox("Communication Method", [
        "Chat Support", 
        "Email Response", 
        "SMS Handling", 
        "Help Center Ticket"
    ])
    
    # Communication-specific inputs
    if communication_method == "Chat Support":
        st.subheader("Chat Support")
        contact_message = st.text_area("Contact Providers")
        
    elif communication_method == "Email Response":
        st.subheader("Email Response")
        email_content = st.text_area("Compose Email")
        recipient = st.selectbox("Select Recipient", 
            [f"{p['Name']} - {p['Medical Spa']}" for p in PROVIDER_DATABASE]
        )
        
    elif communication_method == "SMS Handling":
        st.subheader("SMS Handling")
        sms_message = st.text_area("Send SMS")
        
    elif communication_method == "Help Center Ticket":
        st.subheader("Help Center Ticket")
        ticket_details = st.text_area("Create Help Center Ticket")
        ticket_type = st.selectbox("Ticket Type", [
            "Technical Support",
            "Billing Inquiry",
            "Compliance Question",
            "General Assistance"
        ])
    
    # Send/Create Button
    if st.button("Send/Create"):
        ticket_id = f"MOXIE-{np.random.randint(1000, 9999)}"
        st.success(f"Communication sent/ticket {ticket_id} created successfully!")

# Main Application
def main():
    # Title and Introduction
    st.title("🚀 Moxie AI Support Agent")
    st.markdown("""
        ### Empowering Provider Success Managers
        
        Reduce workload, handle provider interactions efficiently, and focus on critical business challenges.
    """)

    # Tab Navigation
    tab1, tab2, tab3 = st.tabs([
        "AI Chat Support Agent", 
        "Escalation Center", 
        "Insights & Library"
    ])

    with tab1:
        st.header("🤖 Provider Support")
        
        # Question Types
        question_type = st.selectbox("Select Question Type", [
            "Routine Questions",
            "Complex Questions", 
            "Compliance Questions"
        ])
        
        # Provider Question Input
        provider_question = st.text_input("Enter Provider Question")
        
        if provider_question:
            # Generate AI Response
            response, relevant_docs = ask_claude_with_rag(provider_question)
            
            # Display Response
            st.markdown("### 🤖 AI Support Response")
            st.info(response)
            
            # Retrieved Documents
            with st.expander("📚 Relevant Documentation"):
                st.table(relevant_docs)
        
        # Communication Channels
        communication_channels()

    with tab2:
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

    with tab3:
        st.header("📊 Insights & Library")
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions Handled", st.session_state.queries_handled)
        
        with col2:
            st.metric("Escalations", st.session_state.queries_escalated)
        
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
