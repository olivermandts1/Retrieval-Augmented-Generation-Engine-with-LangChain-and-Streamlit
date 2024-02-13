import streamlit as st
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAIChat
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import traceback
import os


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

def show_rag_testing_form():
    TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

    st.title("Retrieval Augmented Generation Engine")

    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="txt", accept_multiple_files=True)

    def load_and_split_documents():
        documents = []
        for source_doc in st.session_state.source_docs:
            text = source_doc.getvalue().decode("utf-8")
            documents.append(Document(text))

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_texts = text_splitter.split_documents(documents)

        # Convert the split documents back into Document objects
        split_doc_objects = [Document(doc.page_content) for doc in split_texts]
        return split_doc_objects

    def embeddings_on_chroma(documents):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents, embedding_function)
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        return retriever

    def query_llm(retriever, query):
        if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
            try:
                # Set the OpenAI API key as an environment variable
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

                # Create an instance of OpenAIChat without passing the API key as a parameter
                llm = OpenAIChat()
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                )
                result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
                
                if result and 'answer' in result:
                    answer = result['answer']
                    st.session_state.messages.append((query, answer))
                    return answer
                else:
                    st.error("No response received or invalid response format.")
                    return None
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return None
        else:
            st.error("OpenAI API key is not set. Please check your Streamlit secrets.")
            return None


    def process_documents():
        if not st.session_state.openai_api_key or not st.session_state.source_docs:
            st.warning("Please upload the documents and provide the missing fields.")
        else:
            try:
                st.info("Loading and splitting documents...")
                documents = load_and_split_documents()
                st.info("Documents loaded and split.")

                st.info("Creating embeddings...")
                st.session_state.retriever = embeddings_on_chroma(documents)
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
