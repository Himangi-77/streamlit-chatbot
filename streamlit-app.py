import streamlit as st
from chat import get_response  # Import the get_response function from chat.py

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #E3F2FD; /* Light sober blue */
        color: #0D47A1; /* Dark blue for text */
    }
    .stTextInput input {
        background-color: #BBDEFB; /* Light blue for input box */
        color: #0D47A1; /* Dark blue for input text */
    }
    .stButton button {
        background-color: #64B5F6; /* Medium blue for button */
        color: white; /* White text for button */
        border: none;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #42A5F5; /* Slightly darker blue on hover */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
        color: #0D47A1; /* Uniform dark blue for headings and text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit UI
st.title("Chat with Sam")
st.markdown("Ask me anything and I'll do my best to help!")

# User input
user_message = st.text_input("You:", placeholder="Type your message here...")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# When the user sends a message
if st.button("Send") and user_message:
    # Add user's message to the chat history
    st.session_state.chat_history.append({"sender": "user", "message": user_message})

    # Get response from the chatbot
    bot_response = get_response(user_message)
    st.session_state.chat_history.append({"sender": "bot", "message": bot_response})

    # Clear the input box
    user_message = ""

# Display chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.write(f"**You:** {chat['message']}")
    else:
        st.write(f"**Sam:** {chat['message']}")
