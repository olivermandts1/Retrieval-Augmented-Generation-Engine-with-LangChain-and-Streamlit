import streamlit as st

from streamlit_gsheets import GSheetsConnection
import pandas as pd


def show_prompt_chain_builder():
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