import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Provider Portal",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
    <style>
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
    
    .connection-options {
        background-color: white;
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ''
if 'show_connection_options' not in st.session_state:
    st.session_state.show_connection_options = False

# Header
st.markdown("""
    <div class='chat-container'>
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Moxie Provider Portal</h1>
            <p style='color: #64748b;'>Your comprehensive healthcare management solution</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Quick Actions with default questions
quick_actions = {
    "📋 Patient Records": "How do I access my patient's medical history?",
    "💉 Treatment Plans": "What are the current treatment protocols?",
    "📱 Mobile App": "How do I use the Moxie mobile app?",
    "📊 Analytics": "Can you show me my practice analytics?"
}

# Display quick actions
cols = st.columns(4)
for i, (action, question) in enumerate(quick_actions.items()):
    with cols[i]:
        if st.button(action, key=f"action_{i}", use_container_width=True):
            st.session_state.chat_input = question
            st.rerun()

# Chat messages display
for message in st.session_state.messages:
    st.markdown(f"""
        <div class='chat-message {message["role"]}'>
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# Chat input and button
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input(
        "",
        value=st.session_state.chat_input,
        placeholder="Type your question here...",
        key="chat_input_field"
    )

with col2:
    if st.button("Searching for your answer...", type="primary", use_container_width=True):
        if user_input.strip():
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Hardcoded response
            response = """
            **Here's what I found for you:**

            **Document Title:** Healthcare Provider Guide
            **Section:** Patient Management
            **Content:** This guide outlines the steps for managing patient records, scheduling appointments, and accessing treatment protocols.
            
            Would you like me to connect you with a Success Manager for more detailed assistance?
            """
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Clear input
            st.session_state.chat_input = ""
            st.rerun()

# Success Manager Connection
if st.button("Connect with Success Manager", use_container_width=True):
    st.session_state.show_connection_options = True
    st.rerun()

# Show connection options
if st.session_state.show_connection_options:
    st.markdown("<div class='connection-options'>", unsafe_allow_html=True)
    st.markdown("### How would you like to connect?")
    
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
                response = f"You've chosen to connect via {option}. A Success Manager will be with you shortly."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.show_connection_options = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #64748b;'>
        © 2024 Moxie Healthcare Solutions
    </div>
""", unsafe_allow_html=True)
