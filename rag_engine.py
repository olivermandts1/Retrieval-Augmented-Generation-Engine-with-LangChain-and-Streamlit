import os, tempfile
import pinecone
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st

from streamlit_gsheets import GSheetsConnection
import pandas as pd


# Create a connection using Streamlit's experimental connection feature
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# Read data from the Google Sheet
df = conn.read(worksheet="PlutusDataImport", ttl=10)
desired_range = df.iloc[99:124, 0:2]  # Rows 100-124 and columns A-B (0-indexed)

# Hardcoding specific values in column A
desired_range.iloc[0:5, 0] = 'Headlines'      # Rows 100-104
desired_range.iloc[6:11, 0] = 'Primary Text'  # Rows 106-110
desired_range.iloc[12:17, 0] = 'Description'  # Rows 112-116
desired_range.iloc[19:24, 0] = 'Forcekeys'    # Rows 119-123

# Replace NaN values with an empty string
desired_range.fillna('', inplace=True)

# Rename the columns
desired_range.columns = ['Asset Type', 'Creative Text']

# Function to format the values, omitting nulls or empty strings
def format_values(asset_type, texts):
    return '\n'.join([f'{asset_type}: {text}' for text in texts if text])

# Store and format values, omitting nulls or empty strings
headlines = format_values('Headlines', desired_range[desired_range['Asset Type'] == 'Headlines']['Creative Text'].tolist())
primary_text = format_values('Primary Text', desired_range[desired_range['Asset Type'] == 'Primary Text']['Creative Text'].tolist())
descriptions = format_values('Description', desired_range[desired_range['Asset Type'] == 'Description']['Creative Text'].tolist())
forcekeys = format_values('Forcekeys', desired_range[desired_range['Asset Type'] == 'Forcekeys']['Creative Text'].tolist())

# Use st.expander to create a toggle for showing the full table
with st.expander("Show Full Table"):
    # Use st.markdown with HTML and CSS to enable text wrapping inside the expander
    st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        white-space: nowrap;
        text-align: left;
        border: 1px solid black;
        padding: 5px;
    }
    .dataframe th {
        background-color: #f0f0f0;
    }
    .dataframe td {
        min-width: 50px;
        max-width: 700px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the DataFrame with text wrapping inside the expander
    st.markdown(desired_range.to_html(escape=False, index=False), unsafe_allow_html=True)


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever()
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

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else: 
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    #
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    