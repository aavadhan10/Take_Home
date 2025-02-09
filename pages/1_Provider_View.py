import streamlit as st

# Page setup
st.set_page_config(page_title="Provider Portal", page_icon="👤", layout="wide")

# Initialize states
if 'active_view' not in st.session_state:
    st.session_state.active_view = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = None

# CSS for consistent alignment
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Center title */
    h1 {
        text-align: center !important;
        margin-bottom: 2rem !important;
        padding: 1rem 0 !important;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem 2rem;
    }
    
    .info-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Align all elements properly */
    .stButton, .stSelectbox, .stTextInput {
        width: 100%;
    }
    
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
        align-items: stretch;
    }
    
    /* Clean up button styling */
    .stButton > button {
        width: 100%;
        height: 100%;
        white-space: nowrap;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
with st.container():
    st.title("Moxie Provider Portal")
    
    # Quick action buttons - all in one row with consistent spacing
    action_cols = st.columns(4)
    actions = {
        "📋 Patient Records": action_cols[0],
        "💉 Treatment Plans": action_cols[1],
        "📱 Moxie App": action_cols[2],
        "📊 Analytics": action_cols[3]
    }

    for action, col in actions.items():
        with col:
            if st.button(action, use_container_width=True):
                st.session_state.active_view = action
                st.session_state.search_type = None
                st.rerun()

    # Content based on active view
    if st.session_state.active_view:
        st.markdown("---")
        
        info_content = {
            "📋 Patient Records": {
                "title": "Patient Records Dashboard",
                "content": """
                ### Recent Patient Activity
                - Latest appointments: 12 today
                - Pending reviews: 5 records
                - Recent updates: 3 new files
                """
            },
            "💉 Treatment Plans": {
                "title": "Treatment Protocols",
                "content": """
                ### Active Protocols
                - Chronic Care Management
                - Preventive Care Plans
                - Specialist Referrals
                """
            },
            "📱 Moxie App": {
                "title": "Mobile App Status",
                "content": """
                ### System Status
                - App version: 3.2.1
                - Last sync: 5 min ago
                - Connected devices: 2
                """
            },
            "📊 Analytics": {
                "title": "Practice Analytics",
                "content": """
                ### Today's Overview
                - Patient visits: 28
                - New registrations: 3
                - Satisfaction rate: 95%
                """
            }
        }
        
        info = info_content[st.session_state.active_view]
        st.markdown(f"## {info['title']}")
        st.markdown(info['content'])

    # Search section - aligned with buttons above
    st.markdown("---")
    search_cols = st.columns([6, 2, 1])
    
    with search_cols[0]:
        search_input = st.text_input("", placeholder="Type your question here...")
        
    with search_cols[1]:
        search_type = st.selectbox("", 
                                 ["Search for answer", "Connect with provider"],
                                 label_visibility="collapsed")
        
    with search_cols[2]:
        search_button = st.button("Go", use_container_width=True)

    # Handle search/connect
    if search_input and search_button:
        st.session_state.search_type = search_type
        st.session_state.active_view = None
        st.rerun()

    # Show response based on search type
    if st.session_state.search_type:
        st.markdown("---")
        
        if st.session_state.search_type == "Search for answer":
            st.markdown("""
            ### Found Relevant Document
            **Title**: Provider Guidelines 2024
            **Section**: Common Procedures
            """)
        
        else:  # Connect with provider
            st.markdown("### Connect with a Success Manager")
            
            connection_cols = st.columns(4)
            connection_options = {
                "💬 Chat": connection_cols[0],
                "📧 Email": connection_cols[1],
                "📱 SMS": connection_cols[2],
                "❓ Help": connection_cols[3]
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
