import streamlit as st
from chatbot import load_chatbot, ask_question  


st.set_page_config(
    page_title="Spotify Chat Assistant",  
    page_icon="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg",  
    layout="centered",
)

# App theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #191414;  /* Spotify black background */
        color: #FFFFFF;  /* White text */
    }
    .stTextInput label {
        color: #FFFFFF;  /* Spotify white text for input labels */
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


qa_chain, vector_store = load_chatbot()


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for chat in st.session_state["chat_history"]:
    st.write(f"**You**: {chat['question']}")
    st.write(f"**Chatbot**: {chat['answer']}")
    st.write("---")

# User input
user_input = st.text_input("Ask a question:")

if user_input:
    
    response = ask_question(qa_chain, user_input)
    
    # Save the chat history
    st.session_state["chat_history"].append({"question": user_input, "answer": response})

    
    st.write(f"**Chatbot**: {response}")

    
    st.write("Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('👍'):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button('👎'):
            st.warning("Sorry the response was not helpful. Visit the Spotify community for more help (https://community.spotify.com).")

    
    st.write("### Related FAQ:")
    related_faqs = vector_store.similarity_search(user_input, k=1)  

    for idx, faq in enumerate(related_faqs):
        question = faq.page_content.split("Answer:")[0].strip()  
        if st.button(question, key=f"faq_{idx}"):  
            st.write(f"**Answer**: {faq.page_content.split('Answer:')[1].strip()}")  

# Sidebar
with st.sidebar:
    st.header("Get Help with Spotify")
    st.write("Ask about subscriptions, account settings, or billing.")


