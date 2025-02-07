import streamlit as st
from anthropic import Anthropic
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import plotly.express as px

# Initialize Claude client
client = Anthropic(api_key="your_anthropic_api_key")  # Replace with your Anthropic API key

# Load internal documentation from CSV
internal_docs_df = pd.read_csv("internal_docs.csv")

# Load provider queries from CSV (for display purposes)
provider_queries_df = pd.read_csv("provider_queries.csv")

# Step 1: Create embeddings for internal docs
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(internal_docs_df["question"].tolist())

# Step 2: Build a FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Step 3: Retrieve relevant documents
def retrieve_documents(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return internal_docs_df.iloc[indices[0]]

# Step 4: Generate response using RAG
def ask_claude_with_rag(query):
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs["question"] + ": " + relevant_docs["answer"])
    
    # Generate response using Claude
    response = client.messages.create(
        model="claude-3.5-sonnet",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
    )
    return response.content[0].text, relevant_docs

# Streamlit app title
st.title("🚀 Moxie AI Agent for PSMs")
st.markdown("### AI-Powered Support to Reduce Your Workload")

# Sidebar for metrics and feedback
st.sidebar.header("📊 PSM Metrics")
queries_handled = st.sidebar.number_input("Queries Handled by AI", value=42)
queries_escalated = st.sidebar.number_input("Queries Escalated to You", value=5)
average_response_time = st.sidebar.text_input("Average Response Time", value="2.3s")

st.sidebar.markdown("---")
st.sidebar.header("🌟 Your Feedback")
feedback = st.sidebar.radio("How is the AI agent helping you?", ["👍 Great", "👎 Needs Improvement"])
if feedback:
    st.sidebar.success("Thank you for your feedback!")

# Main app interface
st.header("🤖 AI Agent Interface")
st.markdown("**Ask the AI agent for help with provider queries.**")

# Display internal docs
with st.expander("📚 View Internal Documentation"):
    st.table(internal_docs_df)

# Display example provider queries
with st.expander("💬 Example Provider Queries"):
    st.table(provider_queries_df)

# PSM query input
psm_query = st.text_input("Ask a question (e.g., 'How do I update billing information?')")

if psm_query:
    # Generate response using RAG
    response, relevant_docs = ask_claude_with_rag(psm_query)
    
    st.markdown("**🤖 AI Agent Response:**")
    st.info(response)
    
    # Show retrieved documents
    with st.expander("🔍 See Relevant Documents Used"):
        st.table(relevant_docs)
    
    # Escalation logic
    if "compliance" in psm_query.lower() or "legal" in psm_query.lower():
        st.warning("🚨 This query has been escalated to you for further review. Ticket ID: #12345")
        queries_escalated += 1
    else:
        queries_handled += 1

# PSM Dashboard
st.header("📈 PSM Dashboard")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Queries Handled by AI", queries_handled)
col2.metric("Queries Escalated to You", queries_escalated)
col3.metric("Average Response Time", average_response_time)

# Visualizations
st.subheader("Query Breakdown")
query_data = {
    "Type": ["Handled by AI", "Escalated to PSM"],
    "Count": [queries_handled, queries_escalated],
}
fig = px.pie(query_data, values="Count", names="Type", title="Query Distribution")
st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using **Claude 3.5 Sonnet**, **Streamlit**, and **RAG**.")
