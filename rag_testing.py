import streamlit as st
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import traceback

def show_rag_testing_form():
    TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

    st.title("Retrieval Augmented Generation Engine")

    # Initialize st.session_state.retriever
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    st.session_state.openai_api_key = st.secrets["openai_secret"]
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="txt", accept_multiple_files=True)

    def load_documents():
        documents = []
        for source_doc in st.session_state.source_docs:
            documents.append(source_doc.getvalue().decode("utf-8"))
        return documents

    def split_documents(documents):
        # Define your chunk size, e.g., number of lines or characters
        chunk_size = 1000  # Example chunk size in characters

        texts = []
        for doc in documents:
            # Split the document into chunks of the specified size
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i+chunk_size]
                texts.append(chunk)
        return texts

    def embeddings_on_chroma(texts):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(texts, embedding_function)
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        return retriever

    def query_llm(retriever, query):
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
        result = result['answer']
        st.session_state.messages.append((query, result))
        return result

    def process_documents():
        if not st.session_state.openai_api_key or not st.session_state.source_docs:
            st.warning("Please upload the documents and provide the missing fields.")
        else:
            try:
                st.info("Loading documents...")
                documents = load_documents()
                st.info("Documents loaded.")

                st.info("Splitting documents...")
                texts = split_documents(documents)
                st.info("Documents split.")

                st.info("Creating embeddings...")
                st.session_state.retriever = embeddings_on_chroma(texts)
                st.info("Embeddings created and retriever initialized.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.text("Error traceback:")
                st.text(traceback.format_exc())

    st.button("Submit Documents", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    
    if query := st.chat_input():
        if st.session_state.retriever is not None:
            st.chat_message("human").write(query)
            response = query_llm(st.session_state.retriever, query)
            st.chat_message("ai").write(response)
        else:
            st.error("Retriever not initialized. Please submit documents first.")

if __name__ == '__main__':
    show_rag_testing_form()

