import streamlit as st

# Page setup
st.set_page_config(page_title="Moxie Provider Portal", page_icon="👤", layout="wide")

# CSS to make it look clean
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    
    h1 {
        text-align: center;
        color: #1e293b;
        padding: 2rem 0;
    }
    
    .stButton > button {
        background-color: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 0.75rem;
        padding: 1.5rem;
        color: #1e293b;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8fafc;
        border: none;
        border-radius: 0.75rem;
        height: 3.5rem;
    }
    
    .stSelectbox > div > div {
        background-color: #f8fafc;
        border: none;
        border-radius: 0.75rem;
        height: 3.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_connect_options' not in st.session_state:
    st.session_state.show_connect_options = False

# Common provider Q&A
PROVIDER_QA = {
    "records": """
        📋 Patient Records Access
        
        • View complete medical history
        • Access test results
        • Update patient information
        • Schedule follow-ups
        
        Need more detailed access? Connect with a Success Manager.
    """,
    "treatment": """
        💉 Treatment Protocols
        
        • Latest clinical guidelines
        • Medication protocols
        • Care pathways
        • Best practices
        
        For specific protocols, check our clinical database.
    """,
    "app": """
        📱 Moxie Mobile App
        
        • Secure patient data access
        • Real-time updates
        • Team communication
        • Schedule management
        
        Download from App Store or Play Store.
    """,
    "analytics": """
        📊 Analytics Dashboard
        
        • Patient demographics
        • Treatment outcomes
        • Practice metrics
        • Financial reports
        
        Custom reports available on request.
    """
}

# Title
st.title("Moxie Provider Portal")

# Quick access buttons
cols = st.columns(4)
buttons = {
    "📋 Patient Records": "records",
    "💉 Treatment Plans": "treatment", 
    "📱 Moxie App": "app",
    "📊 Analytics": "analytics"
}

for i, (label, key) in enumerate(buttons.items()):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.info(PROVIDER_QA[key])

st.markdown("---")

# Search section
search_cols = st.columns([6, 2, 1])
with search_cols[0]:
    question = st.text_input("", placeholder="Type your question here...")
with search_cols[1]:
    action_type = st.selectbox("", 
                              ["Search for answer", "Connect with provider"],
                              label_visibility="collapsed")
with search_cols[2]:
    go_button = st.button("Go", use_container_width=True)

# Handle search/connect
if go_button:
    if action_type == "Search for answer":
        st.info("""
            Here are some relevant resources:
            
            📑 Provider Guidelines 2024
            📚 Clinical Documentation
            🔍 Technical Support
            
            Need more help? You can always connect with a Success Manager.
        """)
    else:
        st.markdown("### Connect with a Success Manager")
        connect_cols = st.columns(4)
        options = ["💬 Chat", "📧 Email", "📱 SMS", "❓ Help"]
        
        for i, option in enumerate(options):
            with connect_cols[i]:
                if st.button(option, key=f"connect_{i}", use_container_width=True):
                    st.success(f"Connecting via {option}... A manager will be with you shortly.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        Moxie Healthcare Solutions
    </div>
""", unsafe_allow_html=True)
