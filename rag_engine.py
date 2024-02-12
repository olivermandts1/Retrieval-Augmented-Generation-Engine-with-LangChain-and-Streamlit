import streamlit as st
import rag_testing
import prompt_chain_builder

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose a menu", ("RAG Testing Form", "Google Sheets Import"))

if choice == "RAG Testing Form":
    rag_testing.show_rag_testing_form()
elif choice == "Google Sheets Import":
    prompt_chain_builder.show_prompt_chain_builder()
