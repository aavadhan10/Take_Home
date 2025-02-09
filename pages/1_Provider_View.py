import streamlit as st

# Page setup
st.set_page_config(page_title="Moxie Provider Portal", page_icon="👤", layout="wide")

# Initialize states
if 'active_view' not in st.session_state:
    st.session_state.active_view = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = None

# CSS for exact matching of design
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Fix button heights and styling */
    .stButton > button {
        background-color: white !important;
        color: #1e293b !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 1.5rem !important;
        height: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    /* Style select box to match */
    .stSelectbox > div > div {
        background-color: #f3f4f6 !important;
        border: none !important;
        border-radius: 0.75rem !important;
        height: 75px !important;
        display: flex !important;
        align-items: center !important;
        min-height: 75px !important;
        font-size: 1rem !important;
    }
    
    /* Style text input to match */
    .stTextInput > div > div > input {
        background-color: rgba(243, 244, 246, 0.7) !important;
        border: none !important;
        border-radius: 0.75rem !important;
        height: 75px !important;
        padding: 0 1.5rem !important;
        font-size: 1rem !important;
        color: #6B7280 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9CA3AF !important;
        opacity: 0.8 !important;
    }
    
    /* Match the dimensions exactly */
    .stTextInput {
        width: 100% !important;
        min-width: 500px !important;
    }
    
    /* Make the button height match */
    .stButton > button {
        height: 75px !important;
        border-radius: 0.75rem !important;
        font-size: 1rem !important;
        padding: 1.5rem !important;
    }
    
    /* Consistent container padding */
    .main .block-container {
        padding: 2rem 5rem !important;
    }
    
    /* Remove default padding from columns */
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0.5rem !important;
    }
    
    /* Title styling */
    h1 {
        font-size: 2.5rem !important;
        color: #1e293b !important;
        margin-bottom: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Moxie Provider Portal")

# Quick action buttons in perfectly aligned white boxes
col1, col2, col3, col4 = st.columns(4)
with col1:
    patient_records = st.button("📋 Patient Records", use_container_width=True)
with col2:
    treatment_plans = st.button("💉 Treatment Plans", use_container_width=True)
with col3:
    moxie_app = st.button("📱 Moxie App", use_container_width=True)
with col4:
    analytics = st.button("📊 Analytics", use_container_width=True)

# Divider
st.markdown("---")

# Search bar row with exact spacing and dimensions
search_cols = st.columns([8, 2, 1])
with search_cols[0]:
    search_input = st.text_input("", placeholder="Type your question here...")
with search_cols[1]:
    search_type = st.selectbox("", 
                             ["Search for answer", "Connect with provider"],
                             label_visibility="collapsed")
with search_cols[2]:
    go_button = st.button("Go", use_container_width=True)

# Divider
st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem;'>
        Moxie Healthcare Solutions
    </div>
""", unsafe_allow_html=True)

# Handle button clicks and show appropriate content (same as before)
if patient_records or treatment_plans or moxie_app or analytics:
    # Show relevant content based on which button was clicked
    pass

if search_input and go_button:
    if search_type == "Search for answer":
        # Show search results
        pass
    else:
        # Show connection options
        connection_cols = st.columns(4)
        with connection_cols[0]:
            st.button("💬 Chat", use_container_width=True)
        with connection_cols[1]:
            st.button("📧 Email", use_container_width=True)
        with connection_cols[2]:
            st.button("📱 SMS", use_container_width=True)
        with connection_cols[3]:
            st.button("❓ Help", use_container_width=True)
