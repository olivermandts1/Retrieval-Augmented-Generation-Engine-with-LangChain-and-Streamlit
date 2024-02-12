import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from openai import OpenAI
import json

def show_content_generator():
    st.subheader("📗 Test Content Generator Prompt Chains")
    st.markdown("Currently Linked CG sheet: https://docs.google.com/spreadsheets/d/1Qim226h-O7xpnqDctwjdmKDwBUGE5rbwELwdjgPeFn0/edit?usp=sharing")
    st.markdown("Prompt Chain Repository: https://docs.google.com/spreadsheets/d/1tqm7G0yzckwSCKXdPcGcWNH6y5nMj68rhpMQZlcO2wU/edit#gid=954337905")
    st.write("#### 1. Retrieved Content Generator Inputs")

    
    # Define the necessary functions
    def replace_dynamic_keys(prompt, topic, value_props, title, intro_paragraph, headers):
        prompt = prompt.replace('[topic]', ', '.join(topic))
        prompt = prompt.replace('[value_props]', ', '.join(value_props))
        prompt = prompt.replace('[title]', ', '.join(title))
        prompt = prompt.replace('[intro_paragraph]', ', '.join(intro_paragraph))
        prompt = prompt.replace('[headers]', ', '.join(headers))
        return prompt

    # Function to generate response using OpenAI API
    def generate_response(system_prompt, user_prompt, model, temperature, api_key, dynamic_values):
        client = OpenAI(api_key=api_key)

        # Ensure prompts are strings
        system_prompt = str(system_prompt) if system_prompt else ""
        user_prompt = str(user_prompt) if user_prompt else ""

        # Replace dynamic keys with actual values
        system_prompt = replace_dynamic_keys(system_prompt, *dynamic_values)
        user_prompt = replace_dynamic_keys(user_prompt, *dynamic_values)

        # Debugging: Print the request payload
        print("Sending request to OpenAI with the following parameters:")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip('"')
        except Exception as e:
            print(f"Error occurred: {e}")
            return "An error occurred while generating the response."
        

    # Create a connection using Streamlit's experimental connection feature
    conn = st.experimental_connection("gsheets", type=GSheetsConnection)

    # Read data from the Google Sheet
    df = conn.read(worksheet="CGDataImport", ttl=10)
    desired_range = df.iloc[0:17, 0:2]  # Rows 100-124 and columns A-B (0-indexed)

    # Hardcoding specific values in column A
    desired_range.iloc[3, 0] = 'Topic'      # Rows 4
    desired_range.iloc[4, 0] = 'Value Props'  # Rows 5
    desired_range.iloc[12, 0] = 'Title'  # Rows 13
    desired_range.iloc[13, 0] = 'Intro Paragraph'    # Rows 14
    desired_range.iloc[14:16, 0] = 'Headers'    # Rows 15-17

    # Replace NaN values with an empty string
    desired_range.fillna('', inplace=True)

    # Rename the columns
    desired_range.columns = ['Asset Type', 'Creative Text']

    # Store values, omitting nulls
    topic = [text for text in desired_range[desired_range['Asset Type'] == 'Topic']['Creative Text'] if text]
    value_props = [text for text in desired_range[desired_range['Asset Type'] == 'Value Props']['Creative Text'] if text]
    title = [text for text in desired_range[desired_range['Asset Type'] == 'Title']['Creative Text'] if text]
    intro_paragraph = [text for text in desired_range[desired_range['Asset Type'] == 'Intro Paragraph']['Creative Text'] if text]
    headers = [text for text in desired_range[desired_range['Asset Type'] == 'Headers']['Creative Text'] if text]
    

    # Use st.expander to create a toggle for showing the full table
    with st.expander("Show Full Table", expanded=True):
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


    # User inputs their OpenAI API key in the sidebar
    openai_api_key = st.secrets["openai_secret"]

        # Dropdown to select a chain name
    chain_names_df = conn.read(worksheet="PromptChainRepo", usecols=['ChainName'], ttl=5)
    chain_names = chain_names_df['ChainName'].dropna().unique().tolist()
    selected_chain = st.selectbox("Select a Prompt Chain", chain_names)

    if st.button("Create Assets"):
        # Fetch the data for the selected chain
        chain_data = conn.read(worksheet="PromptChainRepo", usecols=list(range(40)), ttl=5)
        selected_chain_data = chain_data[chain_data['ChainName'] == selected_chain].iloc[0]

        # Initialize variables for dynamic values
        dynamic_values = (topic, value_props, title, intro_paragraph, headers)

        # Initialize a list to store responses
        st.session_state['responses'] = []

        # Process each link in the chain
        for i in range(1, 11):  # Assuming maximum 10 prompts in a chain
            model_key = f'Model{i}'
            temp_key = f'Temperature{i}'
            sys_prompt_key = f'SystemPrompt{i}'
            user_prompt_key = f'UserPrompt{i}'

            # Check if the entire set of model, temperature, system_prompt, and user_prompt is not null
            if all(pd.notnull(selected_chain_data.get(key)) for key in [model_key, temp_key, sys_prompt_key, user_prompt_key]):
                model = selected_chain_data[model_key]
                temperature = selected_chain_data[temp_key]
                system_prompt = selected_chain_data[sys_prompt_key]
                user_prompt = selected_chain_data[user_prompt_key]

                # Apply dynamic replacements to both system and user prompts
                for j in range(i-1):
                    replacement_text = st.session_state['responses'][j]
                    system_prompt = system_prompt.replace(f'[output {j+1}]', replacement_text)
                    user_prompt = user_prompt.replace(f'[output {j+1}]', replacement_text)

                # Generate response
                response = generate_response(system_prompt, user_prompt, model, temperature, openai_api_key, dynamic_values)
                st.session_state['responses'].append(response)
            else:
                # Stop processing if any of the set is null
                break

        # Display the final response
        if st.session_state['responses']:
            st.write("Debugging Output:", st.session_state['responses'][-1])
                # Check if there are responses and use the latest one

        # Check if there are responses and use the latest one
        if st.session_state.get('responses'):
            json_data = st.session_state['responses'][-1]

            # Convert JSON to dictionary
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                st.error("Invalid JSON format in the response.")
            else:
                # Extracting data from the JSON structure
                assets = data.get("assets", {})
                headlines = [s.split(": ", 1)[1] if ": " in s else s for s in assets.get("headlines", [])[:5]]
                primary_texts = [s.split(": ", 1)[1] if ": " in s else s for s in assets.get("primary_texts", [])[:5]]
                descriptions = [s.split(": ", 1)[1] if ": " in s else s for s in assets.get("descriptions", [])[:5]]

                # Pad lists with blank strings if they have less than 5 elements
                headlines.extend([""] * (5 - len(headlines)))
                primary_texts.extend([""] * (5 - len(primary_texts)))
                descriptions.extend([""] * (5 - len(descriptions)))

                # Create DataFrame
                df = pd.DataFrame({
                    "Content": headlines + [''] + primary_texts + [''] + descriptions
                })

                # Display the DataFrame content for debugging
                st.write("Paste this table into Plutus:")
                st.dataframe(df)

        else:
            st.write("No responses available.")