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
    
    # LLM model from HuggingFace
    llm = HuggingFaceHub(
        repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",  # New conversational model
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    
    # Loading FAQ data
    with open("spotify_faq_data.json") as f:
        faq_data = json.load(f)

    faq_text = [f"{faq['question']} {faq['answer']}" for faq in faq_data]

    # VectorStore and Memory
    vector_store = FAISS.from_texts(faq_text, embedding_model)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Create the conversational chain
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
