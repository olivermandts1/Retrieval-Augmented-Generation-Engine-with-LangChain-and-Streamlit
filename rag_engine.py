import streamlit as st
import rag_testing
import prompt_chain_builder
import creative_text_refresher

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose a menu", ("RAG Testing Form", "Prompt Chain Builder", 'Creative Text Refresher'))

if choice == "RAG Testing Form":
    rag_testing.show_rag_testing_form()
elif choice == "Prompt Chain Builder":
    prompt_chain_builder.show_prompt_chain_builder()
elif choice == 'Creative Text Refresher':
    creative_text_refresher.show_creative_text_refresher()
