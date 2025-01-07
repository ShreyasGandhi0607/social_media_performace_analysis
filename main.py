import streamlit as st
import pandas as pd
from astrapy import DataAPIClient
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Astra DB client
ASTRA_TOKEN = os.getenv('ASTRA_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_KEY')
@st.cache_resource
def init_astra_connection():
    client = DataAPIClient(ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(
        "https://cd4999f4-c8cc-476f-8711-f5f32eecb955-us-east-2.apps.astra.datastax.com"
    )
    collection = db.get_collection("social_engagement")
    return collection

collection = init_astra_connection()

# Fetch Data from Astra DB
@st.cache_data
def fetch_data_from_astra():
    records = list(collection.find({}))  # Fetch all records without filtering
    df = pd.DataFrame(records)
    return df

# GPT Integration for Insights Using Gemini API
def generate_gemini_insights(df, chat_history, api_key):
    """
    Generate insights using Google's Gemini API.

    Parameters:
        df (pd.DataFrame): DataFrame containing engagement data for all post types.
        chat_history (list): List of strings representing chat history.
        api_key (str): Your Google AI Gemini API key.

    Returns:
        str: Generated insights.
    """
    genai.configure(api_key=GEMINI_KEY)
    chat_history_str = "\n".join(chat_history)
    
    # Updated prompt to include all post types data
    prompt = (
        "Analyze the following engagement data across all post types (e.g., carousel, reels, static_image) "
        "and provide insights or comparisons as requested by the user:\n\n"
        f"{df.to_string(index=False)}\n\n"
        f"Chat History:\n{chat_history_str}\n"
        f"User: {chat_history[-1] if chat_history else ''}"
    )
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Social Media Performance Chatbot")

# Initialize session state to preserve chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Text input for user interaction
user_input = st.text_input("Ask something about the data (e.g., comparisons, trends):")

# Handle the chat interaction
if user_input:
    # Add the user's message to the chat history
    st.session_state.chat_history.append(f"User: {user_input}")

    # Fetch all engagement data
    df = fetch_data_from_astra()
    if df.empty:
        st.warning("No data found in the database.")
    else:
        # Generate GPT insights using Gemini API
        api_key = os.getenv("GEMINI_KEY")  # Load your Gemini API key from environment variables
        insights = generate_gemini_insights(df, st.session_state.chat_history, api_key)
        
        # Add the model's response to the chat history
        st.session_state.chat_history.append(f"Model: {insights}")
    
    # Display chat history (user and model messages)
    for message in st.session_state.chat_history:
        st.write(message)
