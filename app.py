import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import load_tools
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from PIL import Image
import io
import base64

# Set the Azure OpenAI API key
os.environ["AZURE_OPENAIAPI_KEY"] = st.secrets["AZURE_OPENAI_KEY"]
# Streamlit app
st.title("Data Analytics Buddy :reminder_ribbon:")
# Catchy description
st.markdown("""
**Welcome to Data Analytics Buddy! 📊**

**Your Intelligent Data Companion:**
Upload your CSV files, ask insightful questions, and get instant answers. Whether it's generating interactive visualizations, performing data analysis, or simply making sense of your data, our AI-powered chatbot is here to assist. Transform your raw data into meaningful insights effortlessly. Let Data Analytics Buddy take your data exploration to the next level!
""")
# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1').fillna(value=0)
    st.write("Uploaded CSV file:")
    st.write(df)
    
    # Initialize the AzureChatOpenAI client
    model = AzureChatOpenAI(
        openai_api_version="2024-05-01-preview",
        azure_deployment="gpt-4o",
        azure_endpoint="https://ai-feedback-system.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAIAPI_KEY")
    )


    # Load the required tools
    # tools = load_tools(["python_repl"], llm=model)

    # Create the agent
    agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

    # Add a text input for user queries
    user_query = st.text_input("Ask a question about your data:")
    
    if user_query:
       # Invoke the agent with the human message and display the output
       response = agent.invoke(user_query)
       st.write("Agent Output:")
       st.write(response)

       # Capture the plot by saving it to a buffer
       buf = io.BytesIO()
       plt.savefig(buf, format='png')
       buf.seek(0)

       # Display the plot
       st.pyplot(plt)

       # Clear the plot to avoid overlapping plots
       plt.clf()

else:
    st.write("Please upload a CSV file to proceed.")
