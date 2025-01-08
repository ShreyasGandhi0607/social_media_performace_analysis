import streamlit as st
import pandas as pd
from astrapy import DataAPIClient
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

@st.cache_resource
def init_astra_connection():
    client = DataAPIClient(ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(
        "https://cd4999f4-c8cc-476f-8711-f5f32eecb955-us-east-2.apps.astra.datastax.com"
    )
    collection = db.get_collection("social_engagement")
    return collection

collection = init_astra_connection()

@st.cache_resource
def fetch_data_from_astra():
    records = list(collection.find({}))
    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()

def generate_gemini_insights(df, chat_history):
    genai.configure(api_key=GEMINI_KEY)
    chat_history_str = "\n".join(chat_history)
    
    prompt = (
        "Analyze the following engagement data and provide insights:\n\n"
        f"{df.head(20).to_string(index=False)}\n\nChat History:\n{chat_history_str}"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating insights: {str(e)}"

st.title("Social Media Performance Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask about the engagement data (e.g., trends):")

if user_input:
    st.session_state.chat_history.append(f"User: {user_input}")
    df = fetch_data_from_astra()
    
    if not df.empty:
        insights = generate_gemini_insights(df, st.session_state.chat_history)
        st.session_state.chat_history.append(f"Model: {insights}")
    else:
        insights = "No data found in the database."
    
    for message in st.session_state.chat_history:
        st.write(message)
