import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Provider View",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (keeping the clean chat interface styling)
st.markdown("""
    <style>
    /* Existing styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        max-width: 85%;
        animation: fade-in 0.3s ease-in-out;
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
    
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
if 'success_manager_mode' not in st.session_state:
    st.session_state.success_manager_mode = False

# Main chat interface with healthcare focus
st.markdown("""
    <div class='chat-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Welcome to Moxie Support</h1>
            <p style='color: #64748b;'>Get instant help with patient care, documentation, and practice management</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Enhanced quick action buttons with healthcare focus
quick_actions = st.columns(4)
actions = [
    "📋 Patient Record Help",
    "💉 Treatment Protocols",
    "📱 Moxie App Support",
    "📊 Practice Analytics"
]

for i, action in enumerate(actions):
    with quick_actions[i]:
        if st.button(action, use_container_width=True):
            st.session_state.chat_input = f"I need help with {action}"

# Chat messages display
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        st.markdown(f"""
            <div class='chat-message {message["role"]}'>
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)

# Chat input
st.markdown("<div style='max-width: 800px; margin: 1rem auto;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([5,1,1])
with col1:
    chat_input = st.text_input(
        "",
        value=st.session_state.chat_input,
        placeholder="Type your question here...",
        key="chat_input_field"
    )

with col2:
    send_button = st.button("Ask a Question", type="primary", use_container_width=True, key="send_button")
    if send_button and chat_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": chat_input
        })
        
        # Check if in success manager mode
        if st.session_state.success_manager_mode:
            response = "Thank you for your message. A Success Manager will review your inquiry and reach out to you directly within 1-2 business hours."
            st.session_state.success_manager_mode = False
        else:
            # Simulate AI response
            response = f"Thank you for your question about {chat_input}. A support specialist will assist you shortly."
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Clear input
        st.session_state.chat_input = ""

with col3:
    success_manager_button = st.button("Connect with Success Manager", type="secondary", use_container_width=True, key="success_manager_button")
    if success_manager_button:
        # Set success manager mode
        st.session_state.success_manager_mode = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "A Moxie Provider Success Manager will be available to assist you shortly. Please describe your inquiry in detail."
        })

st.markdown("</div>", unsafe_allow_html=True)

# Enhanced resources section with healthcare focus
with st.expander("📚 Provider Resources"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            ### Quick Links
            - 📋 Clinical Documentation Guide
            - 💊 Treatment Guidelines
            - 📱 Moxie Mobile App Guide
            - 🏥 Practice Management Tips
        """)
    with col2:
        st.markdown("""
            ### Popular Articles
            - HIPAA Compliance Best Practices
            - Patient Record Management
            - Scheduling System Guide
            - Treatment Planning Tools
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        Moxie Provider Support Portal
    </div>
""", unsafe_allow_html=True)
