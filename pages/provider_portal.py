import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Provider Portal View",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
    <style>
    /* Clean, modern styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Message styling */
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
    
    /* Quick actions */
    .quick-action {
        background-color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .quick-action:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Animations */
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

# Main chat interface
st.markdown("""
    <div class='chat-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Welcome to Moxie Support</h1>
            <p style='color: #64748b;'>How can we help you today?</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Quick action buttons
quick_actions = st.columns(4)
actions = [
    "💳 Billing Help",
    "🔧 Technical Support",
    "📅 Schedule Consultation",
    "📚 Resources"
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
col1, col2 = st.columns([6,1])
with col1:
    chat_input = st.text_input(
        "",
        value=st.session_state.chat_input,
        placeholder="Type your message here...",
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
            
            # Simulate AI response (replace with actual AI integration)
            response = f"Thanks for your message about {chat_input}. A support agent will assist you shortly."
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Clear input
            st.session_state.chat_input = ""
            st.experimental_rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Resources section
with st.expander("📚 Helpful Resources"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### Quick Links
            - [Billing Portal](https://example.com)
            - [Account Settings](https://example.com)
            - [Support Documentation](https://example.com)
            - [Book a Consultation](https://example.com)
        """)
    
    with col2:
        st.markdown("""
            ### Popular Articles
            - Setting up your account
            - Payment methods guide
            - Integration setup
            - Best practices
        """)
