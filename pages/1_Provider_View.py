import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Provider Portal",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with better responsiveness and popup handling
st.markdown("""
    <style>
    /* General styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Enhanced chat styling */
    .chat-container {
        background-color: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        padding: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.75rem;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .chat-message.user {
        background-color: #f1f5f9;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    
    .chat-message.assistant {
        background-color: #0284c7;
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
    }
    
    /* Quick action buttons */
    .quick-action {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .quick-action:hover {
        background-color: #f8fafc;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Connection options styling */
    .connection-options {
        background-color: white;
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .connection-button {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem;
        width: 100%;
        transition: all 0.2s ease;
    }

    .connection-button:hover {
        background-color: #f8fafc;
        transform: translateY(-2px);
    }

    /* Resources section */
    .resources {
        background-color: white;
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .chat-message {
            max-width: 90%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'main'
if 'success_manager_message' not in st.session_state:
    st.session_state.success_manager_message = ''

# Header
st.markdown("""
    <div class='main-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Moxie Provider Portal</h1>
            <p style='color: #64748b;'>Your comprehensive healthcare management solution</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Quick Actions with sample responses
quick_actions = {
    "📋 Patient Records": """
        Here are your recent patient records:
        - Sarah Johnson (Last visit: Today)
        - Michael Chen (Last visit: Yesterday)
        - Emma Davis (Last visit: 2 days ago)
    """,
    "💉 Treatment Plans": """
        Active treatment protocols:
        - Diabetes Management Protocol v2.1
        - Hypertension Care Guidelines 2024
        - Preventive Care Checklist
    """,
    "📱 Mobile App": """
        Moxie Mobile App Status:
        ✅ Connected
        🔄 Last Sync: 5 minutes ago
        📱 Version: 3.2.1
    """,
    "📊 Analytics": """
        Practice Analytics Summary:
        - 95% Patient Satisfaction
        - 28 Appointments Today
        - 3 New Patient Registrations
    """
}

# Display quick actions in a grid
cols = st.columns(4)
for i, (action, response) in enumerate(quick_actions.items()):
    with cols[i]:
        if st.button(action, key=f"action_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": f"I need help with {action}"})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    st.markdown(f"""
        <div class='chat-message {message["role"]}'>
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# Chat input
chat_col1, chat_col2 = st.columns([4, 1])
with chat_col1:
    user_input = st.text_input("", placeholder="Type your question here...", key="chat_input")
with chat_col2:
    send_message = st.button("Send", use_container_width=True)

if send_message and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Sample response based on user input
    response = f"Thank you for your question about '{user_input}'. A provider support specialist will respond shortly."
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Success Manager Connection
if st.button("🤝 Connect with Success Manager", use_container_width=True):
    st.session_state.current_view = 'success_manager'
    st.experimental_rerun()

if st.session_state.current_view == 'success_manager':
    st.markdown("<div class='connection-options'>", unsafe_allow_html=True)
    st.markdown("### Connect with a Success Manager")
    
    message = st.text_area("Your message:", placeholder="Describe how we can help you...")
    
    col1, col2, col3, col4 = st.columns(4)
    connection_options = {
        "💬 Chat": col1,
        "📧 Email": col2,
        "📱 SMS": col3,
        "❓ Help": col4
    }
    
    for option, col in connection_options.items():
        with col:
            if st.button(option, key=f"connect_{option}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"You've chosen to connect via {option}. A Success Manager will contact you shortly.\n\nYour message: {message}"
                })
                st.session_state.current_view = 'main'
                st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Resources Section
with st.expander("📚 Provider Resources", expanded=False):
    st.markdown("<div class='resources'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### Quick Links
            - [Clinical Documentation Guide](#)
            - [Treatment Guidelines](#)
            - [Mobile App Guide](#)
            - [Practice Management Tips](#)
        """)
    
    with col2:
        st.markdown("""
            ### Recent Updates
            - 🆕 Updated HIPAA Guidelines
            - 📱 New Mobile Features
            - 📊 Enhanced Analytics
            - 🏥 Practice Optimization Tools
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        © 2024 Moxie Healthcare Solutions | Built with ❤️ for Healthcare Providers
    </div>
""", unsafe_allow_html=True)
