import streamlit as st
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def show_rag_testing_form():
    TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

    st.title("Retrieval Augmented Generation Engine")


    st.session_state.openai_api_key = st.secrets["openai_secret"]

    # Input fields function
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

    def load_documents():
        loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
        documents = loader.load()
        return documents

    def split_documents(documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
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
            st.warning(f"Please upload the documents and provide the missing fields.")
        else:
            try:
                for source_doc in st.session_state.source_docs:
                    with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                        tmp_file.write(source_doc.read())
                    
                    documents = load_documents()
                    
                    for _file in TMP_DIR.iterdir():
                        temp_file = TMP_DIR.joinpath(_file)
                        temp_file.unlink()
                    
                    texts = split_documents(documents)
                    st.session_state.retriever = embeddings_on_chroma(texts)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.button("Submit Documents", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)
