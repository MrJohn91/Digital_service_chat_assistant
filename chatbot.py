from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import json
import os

def load_chatbot():
    huggingfacehub_api_token = os.getenv("HUGGING_FACE_TOKEN")
    
    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from huggingface_hub import InferenceClient

client = InferenceClient(token=huggingfacehub_api_token)
response = client.text_generation("Your question here", model="google/flan-t5-large")

    # Load the FAQ data
    with open("spotify_faq_data.json") as f:
        faq_data = json.load(f)

    faq_text = [f"{faq['question']} {faq['answer']}" for faq in faq_data]

    vector_store = FAISS.from_texts(faq_text, embedding_model)

    #  Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

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
