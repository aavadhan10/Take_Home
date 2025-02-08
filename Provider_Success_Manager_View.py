import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from anthropic import Anthropic
import json

# Page Configuration
st.set_page_config(
    page_title="Moxie AI Support Agent Demo",
    page_icon="\U0001F680",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

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
        with st.spinner("\U0001F50D Searching through documentation..."):
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

# Risk Analysis Functions
def get_risk_score(query_lower):
    """
    Calculate risk score based on keyword triggers
    """
    escalation_triggers = {
        "critical_risks": {
            "keywords": ["legal action", "lawsuit", "discrimination", 
                        "malpractice", "violation", "harassment", 
                        "unethical", "patient danger"],
            "weight": 3
        },
        "compliance_concerns": {
            "keywords": ["hipaa", "data breach", "confidentiality", 
                        "regulatory violation", "privacy concern", 
                        "medical record", "patient information"],
            "weight": 2
        },
        "urgent_matters": {
            "keywords": ["emergency", "critical", "immediate action", 
                        "urgent resolution", "patient safety", 
                        "life-threatening", "severe incident"],
            "weight": 2
        },
        "technical_issues": {
            "keywords": ["system failure", "integration breakdown", 
                        "security vulnerability", "data loss", 
                        "critical system error"],
            "weight": 1
        }
    }
    
    risk_scores = {
        "critical_risk_score": 0,
        "compliance_risk_score": 0,
        "urgency_score": 0,
        "technical_complexity_score": 0
    }
    
    # Calculate scores for each category
    for category, config in escalation_triggers.items():
        matches = sum(1 for keyword in config["keywords"] if keyword in query_lower)
        if category == "critical_risks":
            risk_scores["critical_risk_score"] = matches * config["weight"] * 10
        elif category == "compliance_concerns":
            risk_scores["compliance_risk_score"] = matches * config["weight"] * 8
        elif category == "urgent_matters":
            risk_scores["urgency_score"] = matches * config["weight"] * 7
        elif category == "technical_issues":
            risk_scores["technical_complexity_score"] = matches * config["weight"] * 5
    
    # Calculate total weighted score (0-100)
    total_score = min(100, sum(risk_scores.values()))
    
    return total_score, risk_scores

def analyze_sentiment_with_claude(query, client):
    """
    Analyzes query sentiment and escalation risk using Claude
    Returns a detailed analysis including sentiment, risk level, and recommended actions
    """
    if client is None:
        return {
            "error": "AI assistant unavailable",
            "risk_level": "Unknown",
            "risk_score": 0,
            "recommended_actions": ["Please check API connection"]
        }
    
    try:
        # Get keyword-based risk score
        query_lower = query.lower()
        keyword_risk_score, risk_breakdown = get_risk_score(query_lower)
        
        prompt = f"""
        Analyze this support query for sentiment, risk level, and needed actions. Provide a structured analysis with:
        1. Overall sentiment (positive, neutral, negative)
        2. Risk level (Low, Medium, High) with clear reasoning
        3. Key concerns identified
        4. Recommended actions
        5. Numerical risk score (0-100)

        Support query: {query}

        Format your response as a JSON object with these exact keys:
        sentiment, risk_level, risk_score, key_concerns, recommended_actions
        """

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            claude_analysis = json.loads(response.content[0].text)
            
            # Combine Claude's risk score with keyword-based score
            combined_risk_score = (keyword_risk_score + claude_analysis["risk_score"]) / 2
            
            analysis = {
                **claude_analysis,
                "risk_score": round(combined_risk_score),
                "risk_breakdown": risk_breakdown,
                "keyword_risk_score": keyword_risk_score,
                "claude_risk_score": claude_analysis["risk_score"]
            }
        except json.JSONDecodeError:
            analysis = {
                "sentiment": "neutral",
                "risk_level": "Medium",
                "risk_score": 50,
                "key_concerns": ["Unable to parse detailed analysis"],
                "recommended_actions": ["Manual review recommended"],
                "risk_breakdown": risk_breakdown,
                "keyword_risk_score": keyword_risk_score,
                "claude_risk_score": 50
            }
            
        # Add color coding for risk levels
        risk_colors = {
            "Low": "#10b981",    # Green
            "Medium": "#f59e0b",  # Yellow
            "High": "#ef4444"     # Red
        }
        
        analysis["risk_color"] = risk_colors.get(analysis["risk_level"], "#64748b")
        
        return analysis
        
    except Exception as e:
        return {
            "error": str(e),
            "risk_level": "Error",
            "risk_score": 0,
            "recommended_actions": ["System error - manual review required"]
        }

def analyze_potential_escalation(query):
    """
    Enhanced escalation analysis using Claude
    """
    sentiment_analysis = analyze_sentiment_with_claude(query, client)
    
    escalation_analysis = {
        "potential_escalation": sentiment_analysis["risk_score"] > 50,
        "risk_level": sentiment_analysis.get("risk_level", "Medium"),
        "risk_score": sentiment_analysis.get("risk_score", 50),
        "detailed_assessment": {
            "Sentiment": sentiment_analysis.get("sentiment", "neutral"),
            "Key Concerns": sentiment_analysis.get("key_concerns", []),
            "risk_breakdown": sentiment_analysis.get("risk_breakdown", {})
        },
        "recommended_actions": sentiment_analysis.get("recommended_actions", []),
        "risk_color": sentiment_analysis.get("risk_color", "#64748b"),
        "keyword_risk_score": sentiment_analysis.get("keyword_risk_score", 0),
        "claude_risk_score": sentiment_analysis.get("claude_risk_score", 0)
    }
    
    return escalation_analysis

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

# Main page layout
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0;'>
        <h1>\U0001F680 Moxie AI Support Agent Demo</h1>
        <p style='color: #64748b;'>Empowering Provider Success Managers with AI assistance</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 AI Support Question Assistant",
    "🚨 Escalation Center & Response Performance Tracker",
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
                
    # Process and display response
    if psm_query:
        response, relevant_docs = ask_claude_with_rag(psm_query)
        
        st.markdown(
            f"""
            <div class='response-container'>
                <h4>🤖 AI Assistant Response</h4>
                <p>{response}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
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
            "response": response
        })

# Tab 2: Escalation Analysis Section
with tab2:
    st.markdown("### 🚨 Escalation Risk Analysis (Powered by an AI Sentiment Analyzer)")
    
    # Escalation Analysis Section
    with st.expander("Analyze Potential Escalation", expanded=True):
        escalation_query = st.text_area(
            "Enter Incident Details", 
            placeholder="Describe the concern or incident in comprehensive detail...",
            height=150
        )
        
        if st.button("🔍 Analyze Escalation Potential", type="primary"):
            if escalation_query:
                analysis = analyze_potential_escalation(escalation_query)
                
                # Display overall risk assessment
                risk_html = f"""
                <div style='
                    background-color: {analysis["risk_color"]}; 
                    color: white; 
                    padding: 15px; 
                    border-radius: 10px;
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <h3 style='margin: 0;'>Risk Level: {analysis["risk_level"]}</h3>
                            <p style='margin: 5px 0 0;'>Risk Score: {analysis["risk_score"]}/100</p>
                        </div>
                        <div style='font-size: 2em;'>
                            {'🚨' if analysis["risk_level"] == "High" else '⚠️' if analysis["risk_level"] == "Medium" else '✅'}
                        </div>
                    </div>
                </div>
                """
                st.markdown(risk_html, unsafe_allow_html=True)
# Tab 2: Escalation Analysis Section
with tab2:
    st.markdown("### 🚨 Escalation Risk Analysis (Powered by an AI Sentiment Analyzer)")
    
    # Escalation Analysis Section
    with st.expander("Analyze Potential Escalation", expanded=True):
        escalation_query = st.text_area(
            "Enter Incident Details", 
            placeholder="Describe the concern or incident in comprehensive detail...",
            height=150
        )
        
        if st.button("🔍 Analyze Escalation Potential", type="primary"):
            if escalation_query:
                analysis = analyze_potential_escalation(escalation_query)
                
                # Display overall risk assessment
                risk_html = f"""
                <div style='
                    background-color: {analysis["risk_color"]}; 
                    color: white; 
                    padding: 15px; 
                    border-radius: 10px;
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <h3 style='margin: 0;'>Risk Level: {analysis["risk_level"]}</h3>
                            <p style='margin: 5px 0 0;'>Risk Score: {analysis["risk_score"]}/100</p>
                        </div>
                        <div style='font-size: 2em;'>
                            {'🚨' if analysis["risk_level"] == "High" else '⚠️' if analysis["risk_level"] == "Medium" else '✅'}
                        </div>
                    </div>
                </div>
                """
                st.markdown(risk_html, unsafe_allow_html=True)
                
                # Display risk score breakdown
                st.subheader("Risk Score Analysis")
                score_cols = st.columns(3)
                with score_cols[0]:
                    st.metric("Combined Risk Score", f"{analysis['risk_score']}/100")
                with score_cols[1]:
                    st.metric("Keyword Risk Score", f"{analysis['keyword_risk_score']}/100")
                with score_cols[2]:
                    st.metric("Claude Risk Score", f"{analysis['claude_risk_score']}/100")
                
                # Display risk breakdown
                st.subheader("Risk Category Breakdown")
                if "risk_breakdown" in analysis["detailed_assessment"]:
                    for category, score in analysis["detailed_assessment"]["risk_breakdown"].items():
                        if score > 0:
                            st.progress(score/100, text=f"{category}: {score}/100")
                
                # Display sentiment and concerns
                st.subheader("Detailed Assessment")
                st.write(f"**Sentiment:** {analysis['detailed_assessment']['Sentiment']}")
                st.write("**Key Concerns:**")
                for concern in analysis["detailed_assessment"]["Key Concerns"]:
                    st.write(f"- {concern}")
                
                # Display recommended actions
                st.subheader("Recommended Actions")
                for action in analysis["recommended_actions"]:
                    st.write(f"- {action}")
            else:
                st.warning("Please enter details for escalation analysis")
    
    # Performance Metrics
    st.markdown("### 📊 Response Performance & Tracker")
    
    # Metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Interactions", "247")
    with col2:
        st.metric("Successful Resolutions", "221 (89.5%)")
    with col3:
        st.metric("Avg Response Time", "12 sec")
    
    # Interaction History
    st.subheader("Interaction Log")
    
    interactions = [
        {
            "timestamp": "2024-02-05 10:23",
            "type": "Billing Query",
            "query": "How to update patient billing?",
            "status": "Resolved",
            "accuracy": "95%"
        },
        {
            "timestamp": "2024-02-05 11:45",
            "type": "Compliance Issue",
            "query": "HIPAA data transfer concern",
            "status": "Escalated",
            "accuracy": "100%"
        }
    ]
    
    for interaction in interactions:
        status_color = "#10b981" if interaction["status"] == "Resolved" else "#ef4444"
        st.markdown(
            f"""
            <div style='
                background-color: #f8fafc; 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
                padding: 15px; 
                margin-bottom: 10px;
            '>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{interaction['timestamp']}</span>
                    <span style='
                        background-color: {status_color};
                        color: white;
                        padding: 3px 8px;
                        border-radius: 4px;
                    '>
                        {interaction['status']}
                    </span>
                </div>
                <p><strong>Type:</strong> {interaction['type']}</p>
                <p><strong>Query:</strong> {interaction['query']}</p>
                <p><strong>Accuracy:</strong> {interaction['accuracy']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Tab 3: Knowledge Base & Interactions
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
    if st.session_state.message_history:
        st.subheader("Recent Messages")
        for msg in reversed(st.session_state.message_history[-5:]):
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p><strong>To:</strong> {msg['provider']}</p>
                    <p><strong>Channel:</strong> {msg['channel']}</p>
                    <p><strong>Message:</strong> {msg['message']}</p>
                    <p><small>Sent: {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Chat History
    st.subheader("Recent AI Interactions")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history[-5:]:  # Show last 5 interactions
            st.markdown(
                f"""
                <div class='metric-card'>
                    <p><strong>Question:</strong> {chat['query']}</p>
                    <p><strong>Response:</strong> {chat['response'][:200]}...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
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
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Built by Ankita Avadhani using Claude 3.5 Sonnet, Streamlit, and RAG
    </div>
    """,
    unsafe_allow_html=True
)
