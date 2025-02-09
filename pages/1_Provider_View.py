import streamlit as st
import time

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
    
    .stTextArea > div > div > textarea {
        background-color: #f8fafc;
        border: none;
        border-radius: 0.75rem;
        padding: 1rem;
    }

    .popup-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 999;
    }

    .popup-content {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 1000;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = None
if 'selected_channel' not in st.session_state:
    st.session_state.selected_channel = None
if 'message_sent' not in st.session_state:
    st.session_state.message_sent = False
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False

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
            st.session_state.current_view = key
            st.session_state.selected_channel = None
            st.session_state.message_sent = False
            st.rerun()

if st.session_state.current_view in PROVIDER_QA:
    st.info(PROVIDER_QA[st.session_state.current_view])

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
    st.session_state.current_view = action_type
    st.session_state.selected_channel = None
    st.session_state.message_sent = False
    st.rerun()

# Show appropriate content based on current view
if st.session_state.current_view == "Search for answer":
    st.info("""
        Here are some relevant resources:
        
        📑 Provider Guidelines 2024
        📚 Clinical Documentation
        🔍 Technical Support
        
        Need more help? You can always connect with a Success Manager.
    """)
elif st.session_state.current_view == "Connect with provider":
    st.markdown("### Connect with a Success Manager")
    
    # Show connection options first
    if not st.session_state.selected_channel:
        st.markdown("### Choose how to connect:")
        connect_cols = st.columns(4)
        options = ["💬 Chat", "📧 Email", "📱 SMS", "❓ Help"]
        
        for i, option in enumerate(options):
            with connect_cols[i]:
                if st.button(option, key=f"connect_{i}", use_container_width=True):
                    st.session_state.selected_channel = option
                    st.rerun()
    
    # After channel is selected, show message input
    if st.session_state.selected_channel and not st.session_state.message_sent:
        st.markdown(f"### Send message via {st.session_state.selected_channel}")
        provider_message = st.text_area(
            "Type your message:", 
            placeholder="Describe what you need help with...",
            height=100
        )
        
        # Send button
        if provider_message and st.button("Send Message", use_container_width=True):
            # Show popup
            popup_html = """
                <div class="popup-overlay"></div>
                <div class="popup-content">
                    <h2 style="margin-bottom: 1rem; color: #1e293b;">Message Sent! ✨</h2>
                    <p style="color: #4b5563;">A Success Manager will contact you shortly.</p>
                </div>
            """
            st.markdown(popup_html, unsafe_allow_html=True)
            
            # Add delay before redirect
            time.sleep(2)
            
            # Reset states
            st.session_state.message_sent = True
            st.session_state.current_view = None
            st.session_state.selected_channel = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        Moxie Healthcare Solutions
    </div>
""", unsafe_allow_html=True)
