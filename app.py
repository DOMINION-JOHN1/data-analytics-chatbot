import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Set the Azure OpenAI API key
os.environ["AZURE_OPENAIAPI_KEY"] = st.secrets["AZURE_OPENAI_KEY"]


# Streamlit app
st.title("Data Analytics Buddy 📊")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1').fillna(value=0)
    st.write("Uploaded CSV file:")
    st.write(df)
    
    # Initialize the AzureChatOpenAI client
    model = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment="gpt-35-turbo-16k",
        azure_endpoint="https://ai-explore1azureai1486566462541.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAIAPI_KEY")
    )

    # Create the agent
    agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=False, allow_dangerous_code=True, handle_parsing_errors=True)

    if user_query: 
        # Invoke the agent with the human message and display the output
        response = agent.invoke(user_query)
        st.write("Agent Output:")
        st.write(response)

        # Check if the response contains a chart
        if hasattr(response, 'chart') and response.chart is not None:
            st.pyplot(response.chart)
        elif isinstance(response, dict) and 'plot' in response:
            # Assume the response contains plotting instructions
            plot_data = response['plot']
            fig, ax = plt.subplots(figsize=(10, 5))

            # Example: check plot type and render accordingly
            if plot_data['type'] == 'line':
                sns.lineplot(data=plot_data['data'], x=plot_data['x'], y=plot_data['y'], ax=ax)
            elif plot_data['type'] == 'bar':
                sns.barplot(data=plot_data['data'], x=plot_data['x'], y=plot_data['y'], ax=ax)
            elif plot_data['type'] == 'scatter':
                sns.scatterplot(data=plot_data['data'], x=plot_data['x'], y=plot_data['y'], ax=ax)
            elif plot_data['type'] == 'hist':
                sns.histplot(data=plot_data['data'], x=plot_data['x'], kde=True, ax=ax)

            plt.title(plot_data.get('title', 'Plot'))
            st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
