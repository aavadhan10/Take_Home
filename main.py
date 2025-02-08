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
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    
    /* Chat Message Styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        max-width: 80%;
    }
    
    .chat-message.user {
        background-color: #e2e8f0;
        margin-left: auto;
    }
    
    .chat-message.assistant {
        background-color: #0284c7;
        color: white;
        margin-right: auto;
    }
    
    /* Quick Access Styling */
    .quick-access-card {
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .quick-access-card.active {
        border: 2px solid #0284c7;
        transform: translateY(-2px);
    }
    
    /* Provider View Styling */
    .provider-header {
        text-align: center;
        padding: 2rem 0;
        background-color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        return f"Error: {str(e)}", pd.DataFrame()

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
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'selected_quick_access' not in st.session_state:
    st.session_state.selected_quick_access = None

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

# View Switcher at the top
st.markdown("""
    <div style='padding: 1rem 0 2rem 0; text-align: center;'>
        <h1 style='color: #0f172a; margin-bottom: 1rem;'>🚀 Moxie AI Support</h1>
    </div>
""", unsafe_allow_html=True)

# Centered columns for the view switcher
col1, col2, col3 = st.columns([2,1,2])
with col2:
    view = st.select_slider(
        "",
        options=["👨‍💼 PSM Dashboard", "👤 Provider Portal"],
        value="👨‍💼 PSM Dashboard",
        key="view_selector"
    )
    st.session_state.current_view = 'PSM' if view == "👨‍💼 PSM Dashboard" else 'Provider'

# Main Content Based on View
if st.session_state.current_view == 'PSM':
    # PSM View - Original tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 AI Support Question Assistant",
        "🚨Escalation Center & Response Performance Tracker",
        "📊 Internal Documentation Search"
    ])
    
    # Tab 1: AI Support Question Assistant
    with tab1:
        st.markdown("### Type in your questions below")
        st.info("Answering common provider questions from internal documentation.")
        
        # Search Section
        psm_query = st.text_input("", placeholder="Type your question here...", key="main_search")
        
        # Center the button
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

        # Process Query and Display Response
        if psm_query:
            response, relevant_docs = ask_claude_with_rag(psm_query)
            
            # Display Response
            st.markdown(f"""
                <div class='response-container'>
                    <h4>🤖 AI Assistant Response</h4>
                    <p>{response}</p>
                </div>
            """, unsafe_allow_html=True)
            
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
                "timestamp": pd.Timestamp.now()
            })

    # Tab 2: Escalation Center & Response Performance
    with tab2:
        st.markdown("### 🚨 Escalation Risk Analysis (Powered by AI Sentiment Analyzer)")
        
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
                    
                    # Display risk assessment
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
                    
                    # Display recommended actions
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
                else:
                    st.warning("Please enter details for escalation analysis")
        
        # Performance Metrics
        st.markdown("### 📊 Response Performance & Tracker")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", st.session_state.queries_handled)
        with col2:
            resolution_rate = (st.session_state.queries_handled - st.session_state.queries_escalated) / max(st.session_state.queries_handled, 1) * 100
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        with col3:
            st.metric("Escalated Cases", st.session_state.queries_escalated)
        
        # Recent Interactions Log
        st.subheader("Recent Interactions")
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history[-5:]):
                st.markdown(f"""
                    <div class='metric-card'>
                        <p><strong>Timestamp:</strong> {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>Query:</strong> {chat['query']}</p>
                        <p><strong>Response:</strong> {chat['response'][:200]}...</p>
                    </div>
                """, unsafe_allow_html=True)

    # Tab 3: Internal Documentation Search
    with tab3:
        st.markdown("### 📊 Knowledge Base & Interactions")
        
        # Document Search
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
        
        # Message History
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

else:
    # Provider Portal View
    st.markdown("""
        <div class='provider-header'>
            <h1 style='color: #0f172a; margin-bottom: 10px;'>Welcome to Moxie Support</h1>
            <p style='color: #64748b; font-size: 1.2em;'>How can we help you today?</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Action Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chat_card = st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2em; margin-bottom: 10px;'>💬</div>
                <h3 style='margin-bottom: 10px;'>Chat Support</h3>
                <p style='color: #64748b;'>Get instant help from our AI assistant</p>
            </div>
        """, unsafe_allow_html=True)
        if chat_card:
            st.session_state.chat_active = not st.session_state.chat_active
    
    with col2:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2em; margin-bottom: 10px;'>❓</div>
                <h3 style='margin-bottom: 10px;'>Help Center</h3>
                <p style='color: #64748b;'>Browse our knowledge base</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 2em; margin-bottom: 10px;'>📞</div>
                <h3 style='margin-bottom: 10px;'>Contact Support</h3>
                <p style='color: #64748b;'>Reach our support team</p>
            </div>
        """, unsafe_allow_html=True)

    # Chat Interface (in sidebar when active)
    if st.session_state.chat_active:
        with st.sidebar:
            st.markdown("### 💬 Chat Support")
            st.markdown("---")
            
            # Chat History
            for msg in st.session_state.chat_messages:
                st.markdown(f"""
                    <div class='chat-message {msg["role"]}'>
                        {msg["content"]}
                    </div>
                """, unsafe_allow_html=True)
            
            # Chat Input
            with st.form("chat_input", clear_on_submit=True):
                user_input = st.text_input("Type your message...", key="chat_input")
                if st.form_submit_button("Send"):
                    if user_input:
                        # Add user message
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": user_input
                        })
                        # Simulate AI response
                        ai_response = "I understand you're asking about " + user_input[:20] + "... Let me help you with that."
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": ai_response
                        })
                        st.experimental_rerun()

    # Quick Access Section
    st.markdown("### 🔗 Quick Access")
    quick_access = st.columns(4)
    
    quick_links = [
        ("💳 Billing", "Update payment info, view invoices", [
            "Update payment method",
            "View recent invoices",
            "Billing history",
            "Payment settings"
        ]),
        ("⚙️ Settings", "Account preferences, notifications", [
            "Profile settings",
            "Notification preferences",
            "Security settings",
            "Integration settings"
        ]),
        ("📊 Analytics", "View business metrics", [
            "Performance dashboard",
            "Revenue analytics",
            "Customer insights",
            "Growth metrics"
        ]),
        ("📚 Resources", "Guidelines, documentation", [
            "User guides",
            "API documentation",
            "Best practices",
            "Video tutorials"
        ])
    ]
    
    # Display Quick Access Cards
    for i, (title, desc, options) in enumerate(quick_links):
        with quick_access[i]:
            card_class = 'active' if st.session_state.selected_quick_access == i else ''
            if st.markdown(f"""
                <div class='metric-card quick-access-card {card_class}' style='text-align: center;'>
                    <h4>{title}</h4>
                    <p style='font-size: 0.9em; color: #64748b;'>{desc}</p>
                </div>
            """, unsafe_allow_html=True):
                st.session_state.selected_quick_access = i

    # Display options for selected quick access card
    if st.session_state.selected_quick_access is not None:
        selected_options = quick_links[st.session_state.selected_quick_access][2]
        st.markdown("---")
        option_cols = st.columns(len(selected_options))
        for i, option in enumerate(selected_options):
            with option_cols[i]:
                st.markdown(f"""
                    <div class='metric-card' style='text-align: center; padding: 10px;'>
                        <p>{option}</p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Built with ❤️ using Claude 3.5 Sonnet, Streamlit, and RAG
    </div>
""", unsafe_allow_html=True)
