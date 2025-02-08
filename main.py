import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic

# Page Configuration
st.set_page_config(
    page_title="Moxie AI Support Agent Demo",
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
    
    /* View Toggle Styling */
    .view-toggle-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    
    .toggle-button {
        background-color: #f1f5f9;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        min-width: 150px;
    }
    
    .toggle-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .toggle-button.active {
        background-color: #0284c7;
        color: white;
        border-color: #0284c7;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Provider Chat Styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        max-width: 85%;
    }
    
    .chat-message.user {
        background-color: #f1f5f9;
        margin-left: auto;
    }
    
    .chat-message.assistant {
        background-color: #0284c7;
        color: white;
        margin-right: auto;
    }
    
    .chat-input {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Response Container */
    .response-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 20px 0;
    }
    
    /* Quick Access Cards */
    .quick-access {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .quick-access-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .quick-access-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
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
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'PSM'
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'selected_quick_access' not in st.session_state:
    st.session_state.selected_quick_access = None
if 'provider_chat_input' not in st.session_state:
    st.session_state.provider_chat_input = ""

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
        return f"Error: {str(e)}", pd.DataFrame()

# Sentiment Analysis and Escalation Function
def analyze_potential_escalation(query):
    escalation_triggers = {
        "critical_risks": [
            "legal action", "lawsuit", "discrimination", 
            "malpractice", "violation", "harassment", 
            "unethical", "patient danger"
        ],
        "compliance_concerns": [
            "hipaa", "data breach", "confidentiality", 
            "regulatory violation", "privacy concern", 
            "medical record", "patient information"
        ],
        "urgent_matters": [
            "emergency", "critical", "immediate action", 
            "urgent resolution", "patient safety", 
            "life-threatening", "severe incident"
        ],
        "technical_issues": [
            "system failure", "integration breakdown", 
            "security vulnerability", "data loss", 
            "critical system error"
        ]
    }
    
    risk_assessment = {
        "critical_risk_score": 0,
        "compliance_risk_score": 0,
        "urgency_score": 0,
        "technical_complexity_score": 0
    }
    
    query_lower = query.lower()
    
    for category, triggers in escalation_triggers.items():
        category_matches = [trigger for trigger in triggers if trigger in query_lower]
        
        if category == "critical_risks":
            risk_assessment["critical_risk_score"] = len(category_matches) * 3
        elif category == "compliance_concerns":
            risk_assessment["compliance_risk_score"] = len(category_matches) * 2
        elif category == "urgent_matters":
            risk_assessment["urgency_score"] = len(category_matches) * 2
        elif category == "technical_issues":
            risk_assessment["technical_complexity_score"] = len(category_matches) * 2
    
    total_risk_score = (
        risk_assessment["critical_risk_score"] * 3 +
        risk_assessment["compliance_risk_score"] * 2 +
        risk_assessment["urgency_score"] * 2 +
        risk_assessment["technical_complexity_score"]
    )
    
    escalation_analysis = {
        "potential_escalation": total_risk_score > 5,
        "risk_level": "High" if total_risk_score > 10 else "Medium" if total_risk_score > 5 else "Low",
        "risk_score": total_risk_score,
        "detailed_assessment": risk_assessment,
        "recommended_actions": []
    }
    
    if total_risk_score > 10:
        escalation_analysis["recommended_actions"].append("🚨 Immediate Management Review Required")
    elif total_risk_score > 5:
        escalation_analysis["recommended_actions"].append("⚠️ Consultation with Senior Management Recommended")
    
    if risk_assessment["critical_risk_score"] > 0:
        escalation_analysis["recommended_actions"].append("🛡️ Engage Legal Department")
    
    if risk_assessment["compliance_risk_score"] > 0:
        escalation_analysis["recommended_actions"].append("📋 Compliance Team Review")
    
    if risk_assessment["urgency_score"] > 0:
        escalation_analysis["recommended_actions"].append("⏰ Prioritize Immediate Response")
    
    if risk_assessment["technical_complexity_score"] > 0:
        escalation_analysis["recommended_actions"].append("🖥️ Technical Support Consultation")
    
    return escalation_analysis

# View Toggle at the top
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #0f172a; margin-bottom: 1rem;'>🚀 Moxie AI Support</h1>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2,1,2])
with col2:
    view_col1, view_col2 = st.columns(2)
    with view_col1:
        if st.button("👨‍💼 PSM View", key="psm_view", type="primary" if st.session_state.current_view == 'PSM' else "secondary", use_container_width=True):
            st.session_state.current_view = 'PSM'
    with view_col2:
        if st.button("👤 Provider View", key="provider_view", type="primary" if st.session_state.current_view == 'Provider' else "secondary", use_container_width=True):
            st.session_state.current_view = 'Provider'
# Main Content Based on View
if st.session_state.current_view == 'PSM':
    # PSM View with sidebar navigation
    st.sidebar.markdown("### PSM Navigation")
    st.sidebar.markdown("---")
    
    # Performance Metrics in Sidebar
    st.sidebar.markdown("### Performance Metrics")
    metrics_cols = st.sidebar.columns(2)
    with metrics_cols[0]:
        st.metric("Queries Handled", st.session_state.queries_handled)
    with metrics_cols[1]:
        st.metric("Escalated", st.session_state.queries_escalated)
    
    # Main PSM Content with tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 AI Support Question Assistant",
        "🚨Escalation Center",
        "📊 Documentation Search"
    ])
    
    with tab1:
        st.markdown("### Type in your questions below")
        st.info("Answering common provider questions from internal documentation.")
        
        psm_query = st.text_input("", placeholder="Type your question here...", key="psm_search")
        
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
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

        if psm_query:
            response, relevant_docs = ask_claude_with_rag(psm_query)
            
            st.markdown(f"""
                <div class='response-container'>
                    <h4>🤖 AI Assistant Response</h4>
                    <p>{response}</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📚 Relevant Internal Documentation"):
                st.dataframe(
                    relevant_docs[["question", "answer"]],
                    use_container_width=True,
                    column_config={
                        "question": "Question",
                        "answer": "Answer"
                    }
                )
            
            st.session_state.queries_handled += 1
            st.session_state.chat_history.append({
                "query": psm_query,
                "response": response,
                "timestamp": pd.Timestamp.now()
            })

    with tab2:
        st.markdown("### 🚨 Escalation Analysis")
        
        # Escalation Analysis Section
        with st.expander("Analyze Potential Escalation", expanded=True):
            escalation_query = st.text_area(
                "Enter Incident Details", 
                placeholder="Describe the concern or incident in comprehensive detail...",
                height=150
            )
            
            if st.button("🔍 Analyze Escalation Potential", type="primary"):
                if escalation_query:
                    escalation_analysis = analyze_potential_escalation(escalation_query)
                    
                    # Display risk level
                    risk_color = {
                        "Low": "#10b981",
                        "Medium": "#f59e0b",
                        "High": "#ef4444"
                    }[escalation_analysis["risk_level"]]
                    
                    st.markdown(f"""
                        <div style='
                            background-color: {risk_color}; 
                            color: white; 
                            padding: 15px; 
                            border-radius: 10px;
                            margin-bottom: 20px;
                        '>
                            <h3 style='margin: 0;'>Risk Level: {escalation_analysis["risk_level"]}</h3>
                            <p style='margin: 5px 0 0;'>Risk Score: {escalation_analysis["risk_score"]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommended Actions
                    st.markdown("### Recommended Actions:")
                    for action in escalation_analysis["recommended_actions"]:
                        st.markdown(f"- {action}")
                    
                    # Detailed Assessment
                    st.markdown("### Detailed Risk Assessment:")
                    risk_df = pd.DataFrame({
                        "Risk Category": escalation_analysis["detailed_assessment"].keys(),
                        "Score": escalation_analysis["detailed_assessment"].values()
                    })
                    st.dataframe(risk_df)
                    
                    if escalation_analysis["risk_level"] != "Low":
                        st.session_state.queries_escalated += 1
                else:
                    st.warning("Please enter details for escalation analysis")
        
        # Performance Metrics
        st.markdown("### 📊 Response Performance & Tracker")
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Total Interactions", st.session_state.queries_handled)
        with metrics_cols[1]:
            resolution_rate = (st.session_state.queries_handled - st.session_state.queries_escalated) / max(st.session_state.queries_handled, 1) * 100
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        with metrics_cols[2]:
            st.metric("Escalated Cases", st.session_state.queries_escalated)

    with tab3:
        st.markdown("### 📊 Documentation Search")
        
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

else:
    # Provider View (ChatGPT style)
    st.markdown("""
        <div class="chat-container">
            <div style="text-align: center; margin-bottom: 1rem;">
                <h2>Moxie Support Assistant</h2>
                <p style="color: #64748b;">How can I help you today?</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick Access Suggestions
    quick_access = st.columns(4)
    suggestions = [
        "💳 Billing Support",
        "🔧 Technical Help",
        "📱 Account Settings",
        "📚 Resources"
    ]
    
    for i, suggestion in enumerate(suggestions):
        with quick_access[i]:
            if st.button(suggestion, use_container_width=True):
                st.session_state.provider_chat_input = f"I need help with {suggestion}"
    
    # Chat Interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            st.markdown(f"""
                <div class="chat-message {message['role']}">
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat Input
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    col1, col2 = st.columns([6,1])
    with col1:
        chat_input = st.text_input(
            "",
            value=st.session_state.provider_chat_input,
            placeholder="Type your message here...",
            key="provider_chat"
        )
    with col2:
        if st.button("Send", type="primary", use_container_width=True):
            if chat_input:
                # Add user message
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": chat_input
                })
                
                # Get AI response using RAG
                response, _ = ask_claude_with_rag(chat_input)
                
                # Add AI response
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Clear input
                st.session_state.provider_chat_input = ""
                st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Additional Resources (collapsed by default)
    with st.expander("📚 Additional Resources"):
        resources_col1, resources_col2 = st.columns(2)
        with resources_col1:
            st.markdown("""
                ### Quick Links
                - 📋 Billing & Payments
                - 🔐 Account Security
                - 📱 Mobile App Guide
                - 📞 Contact Support
            """)
        with resources_col2:
            st.markdown("""
                ### Popular Articles
                - How to update payment methods
                - Setting up 2FA
                - Integration guides
                - Best practices
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Built with ❤️ using Claude 3.5 Sonnet, Streamlit, and RAG
    </div>
""", unsafe_allow_html=True)
