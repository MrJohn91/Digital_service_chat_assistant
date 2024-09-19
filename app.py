import streamlit as st
from chatbot import load_chatbot, ask_question  

# page config
st.set_page_config(
    page_title="Spotify Chat Assistant",  
    page_icon="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg",  
    layout="centered",
)

# app theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #191414;  /* Spotify black background */
        color: #FFFFFF;  /* White text */
    }
    .stTextInput label {
        color: #1DB954;  /* Spotify green for input labels */
    }
    .stButton button {
        background-color: #1DB954;  /* Spotify green for button */
        color: #FFFFFF;  /* White text on button */
    }
    .stTextInput input {
        background-color: #262626;  /* Darker input box */
        color: #FFFFFF;  /* White text in input */
    }
    .stMarkdown h1 {
        color: #1ED760;  /* Spotify green for the title */
    }
    </style>
    """, unsafe_allow_html=True
)

# Streamlit UI 
st.title("Spotify Chat Assistant 🎧")

# Load the chatbot
qa_chain, vector_store = load_chatbot()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Ask a question:")

if user_input:
    # Get chatbot response
    response = ask_question(qa_chain, user_input)
    st.session_state["chat_history"].append({"question": user_input, "answer": response})
    st.write(f"**Chatbot**: {response}")


    # Display the feedback options only after a response
    st.write("Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('👍'):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button('👎'):
            st.warning("Sorry the response was not helpful. Visit the Spotify community for more help (https://community.spotify.com).")


    # Display one related FAQ suggestion (without showing the answer)
    st.write("### Related FAQ:")
    related_faqs = vector_store.similarity_search(user_input, k=1)  # Show only 1 related FAQ

    # Only show the question, not the answer
    for idx, faq in enumerate(related_faqs):
        question = faq.page_content.split("Answer:")[0].strip()  # Extract the question part
        if st.button(question, key=f"faq_{idx}"):  
            st.write(f"**Answer**: {faq.page_content.split('Answer:')[1].strip()}")  # Show answer only on click

# Sidebar with FAQ topics
with st.sidebar:
    st.header("How to use")
    st.write("Ask the chatbot about Spotify subscriptions, account settings, or billing.")
    topic = st.selectbox("Choose a topic:", ["App Help", "Payment Help", "Device Help", "Account Help"])
    st.write("Topic selected:", topic)



