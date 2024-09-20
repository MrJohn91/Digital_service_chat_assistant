import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import json

def load_chatbot():
    # Ensure OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OpenAI API key not found.")

    # Initialize the OpenAI Chat Model (gpt-4 or gpt-3.5-turbo)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # Embedding model for retrieval
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAQ data
    with open("spotify_faq_data.json") as f:
        faq_data = json.load(f)

    faq_text = [f"{faq['question']} {faq['answer']}" for faq in faq_data]

    # Create FAISS vector store
    vector_store = FAISS.from_texts(faq_text, embedding_model)

    # Create memory for conversational context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Set up Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Use OpenAI LLM from ChatOpenAI class
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True
    )

    return qa_chain, vector_store


def ask_question(qa_chain, query):
    try:
        result = qa_chain({"question": query})
        return result["answer"]
    except openai.error.OpenAIError as e:
        # Handle OpenAI API-specific errors
        return f"An error occurred: {str(e)}"
    except Exception as e:
        # Handle other generic errors
        return f"An unexpected error occurred: {str(e)}"
