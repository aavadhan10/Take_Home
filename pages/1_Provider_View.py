import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Provider View",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (improved alignment and styling)
st.markdown("""
    <style>
    /* General styling */
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

    /* Hide Streamlit default margins and paddings */
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }

    /* Improved alignment for buttons and inputs */
    .stButton button {
        width: 100%;
    }

    .stTextInput input {
        width: 100%;
    }

    /* Connection options styling */
    .connection-options {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 1rem;
    }

    .connection-options button {
        width: 100%;
        text-align: left;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        background-color: white;
        color: #333;
    }

    .connection-options button:hover {
        background-color: #f1f5f9;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with default values
def initialize_session_state():
    default_states = {
        'messages': [],
        'chat_input': '',
        'success_manager_mode': False,
        'show_connection_options': False,
        'success_manager_message': ''  # New state for success manager message
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call initialization
initialize_session_state()

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
for message in st.session_state.messages:
    st.markdown(f"""
        <div class='chat-message {message["role"]}'>
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# Chat input and buttons
st.markdown("<div style='max-width: 800px; margin: 1rem auto;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([5,1,1])

with col1:
    # Text input for chat
    chat_input = st.text_input(
        "",
        value=st.session_state.chat_input,
        placeholder="Type your question here...",
        key="chat_input_field"
    )

with col2:
    # Ask button
    ask_clicked = st.button("Ask", type="primary", use_container_width=True, key="ask_primary_button")

with col3:
    # Success Manager button
    success_manager_clicked = st.button("Connect with a Success Manager", type="secondary", use_container_width=True, key="success_manager_primary_button")

# Handle button clicks
if ask_clicked and chat_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": chat_input
    })
    
    # Hardcoded result
    response = "Searching through relevant documentation..."
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Clear input
    st.session_state.chat_input = ""

if success_manager_clicked:
    # Set success manager mode
    st.session_state.show_connection_options = True

# Show connection options if "Connect with Success Manager" is clicked
if st.session_state.show_connection_options:
    st.markdown("---")
    st.markdown("### How would you like to connect with a Success Manager?")
    
    # Text input for leaving a message
    st.session_state.success_manager_message = st.text_area(
        "Leave a message for the Success Manager (optional):",
        value=st.session_state.success_manager_message,
        placeholder="Describe your inquiry here..."
    )
    
    # Options for connecting
    st.markdown('<div class="connection-options">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("💬 Chat Support", key="chat_support"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"You have chosen to connect via Chat Support. A Success Manager will be with you shortly.\n\nYour message: {st.session_state.success_manager_message}"
            })
            st.session_state.show_connection_options = False  # Close the options
            st.session_state.success_manager_message = ""  # Clear the message
    with col2:
        if st.button("📧 Email", key="email"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"You have chosen to connect via Email. A Success Manager will contact you shortly.\n\nYour message: {st.session_state.success_manager_message}"
            })
            st.session_state.show_connection_options = False  # Close the options
            st.session_state.success_manager_message = ""  # Clear the message
    with col3:
        if st.button("📱 SMS", key="sms"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"You have chosen to connect via SMS. A Success Manager will contact you shortly.\n\nYour message: {st.session_state.success_manager_message}"
            })
            st.session_state.show_connection_options = False  # Close the options
            st.session_state.success_manager_message = ""  # Clear the message
    with col4:
        if st.button("❓ Help Center", key="help_center"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"You have chosen to visit the Help Center. Here's the link: [Help Center](#)\n\nYour message: {st.session_state.success_manager_message}"
            })
            st.session_state.show_connection_options = False  # Close the options
            st.session_state.success_manager_message = ""  # Clear the message
    st.markdown('</div>', unsafe_allow_html=True)

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
