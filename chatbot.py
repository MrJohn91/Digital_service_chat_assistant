import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json
import os

def load_chatbot():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # FAQ data
    with open("spotify_faq_data.json") as f:
        faq_data = json.load(f)

    faq_text = [f"{faq['question']} {faq['answer']}" for faq in faq_data]

    # FAISS VectorStore for retrieval
    vector_store = FAISS.from_texts(faq_text, embedding_model)

    # Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=None,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True
    )

    return qa_chain, vector_store

def ask_question(query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message["content"].strip()
