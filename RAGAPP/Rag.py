import os
from pypdf import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings,ChatHuggingFace
from langchain_community.vectorstores import FAISS



# getting text from PDFs
def get_text(docs):
    text = ""
    for doc in docs:
        pdf = PdfReader(doc)
        for page in pdf.pages:
            text = text+page.extract_text()
    return text

# Chunks of text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators='\n',chunk_size = 100, chunk_overlap=20)
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# Convert text to embeddings and store in faiss db

def text_to_embeddings(chunks):
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        # model= "jinaai/jina-embeddings-v2-base-en",
        model = 'sentence-transformers/paraphrase-MiniLM-L6-v2',
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )

    db = FAISS.from_texts(chunks,hf_embeddings)
    return db


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
        
# builing Gui

def main():

    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    st.set_page_config(page_title="Chat with multiple PDFs")
    st.title("Chat with PDfs")
    st.subheader("Ask Questions on your PDFs")
    question = st.text_input("Please ask question")

    if st.button("Text Generation"):

        context = st.session_state.retriever

        template = """Use only and only the following pieces of context to answer the question at the end.
        If you don't know the answer, just respond you don't know the 'answer', don't hallucinate and try to make up an answer
        Its ok to tell you dont know the answer.
        {context}

        Question: {question}

        Helpful Answer:"""
    
        prompt = PromptTemplate.from_template(template)

        llm = HuggingFaceEndpoint(
            repo_id = 'microsoft/Phi-3-mini-4k-instruct',
            temperature = 0.7,
            huggingfacehub_api_token = os.getenv('HF_TOKEN')
        )

        chain = ({'context':context|format_docs,'question':RunnablePassthrough()} | prompt | llm)

        response = chain.invoke(question)
        
        st.markdown(response)



    with st.sidebar:
        docs = st.file_uploader(label=':books: Upload your Files',accept_multiple_files=True)
        if st.button('process the Docs'):
            with st.spinner():
                
                # convert docs to text
                raw_text = get_text(docs)
                # convert text to chunks
                text_chunks = get_text_chunks(raw_text)
                # convert chunks to embeddings and store in vector database
                vector_db = text_to_embeddings(text_chunks)
                st.session_state.retriever = vector_db.as_retriever()

    


if __name__ == "__main__":
    main()