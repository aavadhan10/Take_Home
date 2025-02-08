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
if 'success_manager_request' not in st.session_state:
    st.session_state.success_manager_request = False

# Main chat interface with healthcare focus
st.markdown("""
    <div class='chat-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Welcome to Moxie Support</h1>
            <p style='color: #64748b;'>Get instant help with patient care, documentation, and practice management</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Quick actions with Success Manager button
st.markdown("### Quick Actions")
quick_actions = st.columns([1,1,1,1,1.5])  # Adjusted column widths
actions = [
    "📋 Patient Record Help",
    "💉 Treatment Protocols",
    "📱 Moxie App Support",
    "📊 Practice Analytics",
    "🤝 Connect with Success Manager"  # Modified button text
]

for i, action in enumerate(actions):
    with quick_actions[i]:
        if st.button(action, use_container_width=True):
            if "Success Manager" in action:
                # Set flag for Success Manager connection
                st.session_state.success_manager_request = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "A Moxie Provider Success Manager will contact you shortly. Please provide a brief description of your inquiry."
                })
                # Optionally, you could add a modal or more prominent UI element here
                st.info("Success Manager connection request initiated. Please describe your inquiry in the chat.")
            else:
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
col1, col2 = st.columns([6,1])
with col1:
    chat_input = st.text_input(
        "",
        value=st.session_state.chat_input,
        placeholder="Type your question here...",
        key="chat_input_field"
    )
with col2:
    if st.button("Send", type="primary", use_container_width=True):
        if chat_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": chat_input
            })
            
            # Check if this is a response to Success Manager request
            if st.session_state.success_manager_request:
                response = f"Thank you for your detailed message. Our Success Manager team will review your inquiry and reach out to you directly. Typical response time is within 1-2 business hours."
                st.session_state.success_manager_request = False
            else:
                # Simulate AI response
                response = f"Thank you for your question about {chat_input}. A support specialist will assist you shortly."
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Clear input
            st.session_state.chat_input = ""
            st.experimental_rerun()
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
