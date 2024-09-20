from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI 
import os
import json

def load_chatbot():
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OpenAI API key not found.")

    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAQ data
    with open("spotify_faq_data.json") as f:
        faq_data = json.load(f)

    faq_text = [f"{faq['question']} {faq['answer']}" for faq in faq_data]

    # FAISS vector store
    vector_store = FAISS.from_texts(faq_text, embedding_model)

    #memory for conversational
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True
    )

    return qa_chain, vector_store

def ask_question(qa_chain, query):
    result = qa_chain({"question": query})
    return result["answer"]
