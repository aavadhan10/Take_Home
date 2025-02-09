import streamlit as st

# Page setup
st.set_page_config(page_title="Provider Portal", page_icon="👤", layout="wide")

# Initialize states
if 'active_view' not in st.session_state:
    st.session_state.active_view = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = None

# CSS for cleaner UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
    }
    
    .info-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .connection-options {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Moxie Provider Portal")

# Quick action buttons
quick_actions = st.columns(4)
actions = [
    "📋 Patient Records",
    "💉 Treatment Plans",
    "📱 Moxie App",
    "📊 Analytics"
]

for i, action in enumerate(actions):
    with quick_actions[i]:
        if st.button(action, use_container_width=True):
            st.session_state.active_view = action
            st.session_state.search_type = None  # Reset search type
            st.rerun()

# Show info based on active view
if st.session_state.active_view:
    st.markdown("---")
    
    # Different content for each tab
    info_content = {
        "📋 Patient Records": {
            "title": "Patient Records Dashboard",
            "content": """
            ### Recent Patient Activity
            - Latest appointments: 12 today
            - Pending reviews: 5 records
            - Recent updates: 3 new files
            
            ### Quick Actions
            - Schedule follow-up
            - Update patient info
            - Access medical history
            """
        },
        "💉 Treatment Plans": {
            "title": "Treatment Protocols",
            "content": """
            ### Active Protocols
            - Chronic Care Management
            - Preventive Care Plans
            - Specialist Referrals
            
            ### Resources
            - Clinical guidelines
            - Treatment templates
            - Care coordination tools
            """
        },
        "📱 Moxie App": {
            "title": "Mobile App Status",
            "content": """
            ### System Status
            - App version: 3.2.1
            - Last sync: 5 min ago
            - Connected devices: 2
            
            ### Features
            - Patient messaging
            - Schedule management
            - Document access
            """
        },
        "📊 Analytics": {
            "title": "Practice Analytics",
            "content": """
            ### Today's Overview
            - Patient visits: 28
            - New registrations: 3
            - Satisfaction rate: 95%
            
            ### Trends
            - Weekly growth: +5%
            - Peak hours: 9-11 AM
            - Popular services
            """
        }
    }
    
    info = info_content[st.session_state.active_view]
    st.markdown(f"## {info['title']}")
    st.markdown(info['content'])

# Search and connect section
st.markdown("---")
col1, col2 = st.columns([4,1])

with col1:
    search_input = st.text_input("", placeholder="Type your question here...")

with col2:
    search_type = st.selectbox("", 
                              ["Search for answer", "Connect with provider"],
                              label_visibility="collapsed")

# Handle search/connect
if search_input and st.button("Go", use_container_width=True):
    st.session_state.search_type = search_type
    st.session_state.active_view = None  # Clear previous view
    st.rerun()

# Show response based on search type
if st.session_state.search_type:
    st.markdown("---")
    
    if st.session_state.search_type == "Search for answer":
        st.markdown("""
        ### Found Relevant Document
        
        **Title**: Provider Guidelines 2024
        **Section**: Common Procedures
        **Summary**: This document outlines standard protocols for...
        
        Would you like to:
        - View full document
        - Get related resources
        - Connect with specialist
        """)
    
    else:  # Connect with provider
        st.markdown("### Connect with a Success Manager")
        
        col1, col2, col3, col4 = st.columns(4)
        connection_options = {
            "💬 Chat": col1,
            "📧 Email": col2,
            "📱 SMS": col3,
            "❓ Help": col4
        }
        
        for option, col in connection_options.items():
            with col:
                if st.button(option, use_container_width=True):
                    st.success(f"Connecting via {option}... A manager will be with you shortly.")
                    st.session_state.search_type = None
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b;'>
        Moxie Healthcare Solutions
    </div>
""", unsafe_allow_html=True)
