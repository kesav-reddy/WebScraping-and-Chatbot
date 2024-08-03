import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import openai  # Import OpenAI directly
from sklearn.metrics.pairwise import cosine_similarity

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["api_key"]
openai.api_key = api_key  # Set the API key for OpenAI

# Load the embedded text 
df = pd.read_csv('embedded_articles.csv')

st.title("Chatbot using GPT-4 about Aluminium Articles")  # Corrected title
st.write("Ask me anything about Aluminium Updates for the past 45 days!")

# Input for the users
user_input = st.text_input("You:", "")

# Function to get a response from GPT-4
def get_response(prompt, context):
    response = openai.ChatCompletion.create(  # Use the OpenAI API directly
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"{context}\n\n{prompt}"}
        ]
    )
    return response.choices[0].message['content'].strip()

# Function to find the most relevant article
def find_most_relevant_article(user_query):
    # Create an embedding for the user's query
    response = openai.Embedding.create(  # Use the OpenAI API directly
        input=user_query,
        model="text-embedding-ada-002"
    )
    
    query_embedding = np.array(response['data'][0]['embedding']) 
    
    embeddings = np.array(df['ada_embedding'].apply(json.loads).tolist())
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    most_similar_index = np.argmax(similarities)
    
    return df.iloc[most_similar_index]

# When the user presses Enter
if user_input:
    try:
        relevant_article = find_most_relevant_article(user_input)
        context = (
            f"You are a knowledgeable assistant about Aluminium articles. "
            f"Here is an article that may be relevant:\n"
            f"Title: {relevant_article['title']}\n"
            f"Description: {relevant_article['description']}\n"
            f"Date: {relevant_article['date']}\n"
            f"Link: {relevant_article['link']}\n"
            "Please provide specific information or answer the user's question based on the article."
        )
        
        response = get_response(user_input, context)
        st.write(f"Bot: {response}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# import os
# import json
# import pandas as pd
# import numpy as np
# import streamlit as st
# from openai import OpenAI
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity

# # # Loading the API key 
# # load_dotenv('API_Key.env')
# # api_key = os.getenv("API_KEY")
# # client = OpenAI(api_key=api_key)
# api_key = st.secrets["api_key"]

# # Create an OpenAI client
# #openai.api_key = api_key
# client = OpenAI(api_key=api_key)


# # Loading the embedded text 
# df = pd.read_csv('embedded_articles.csv')

# st.title("Chatbot using GPT-4o about Aluminium Articles")
# st.write("Ask me anything about Aluminium Updates for the past 45 days!")

# #Input for the users
# user_input = st.text_input("You:", "")

# # Function to get a response from GPT-4o
# def get_response(prompt, context):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "user", "content": f"{context}\n\n{prompt}"}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# # Function to find the most relevant article
# def find_most_relevant_article(user_query):
   
#     response = client.embeddings.create(
#         input=user_query,
#         model="text-embedding-ada-002"
#     )
    
   
#     query_embedding = np.array(response.data[0].embedding) 
    
#     embeddings = np.array(df['ada_embedding'].apply(json.loads).tolist())  # Adjust as needed
    
#     # Calculate cosine similarity
#     similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
#     most_similar_index = np.argmax(similarities)
    
#     return df.iloc[most_similar_index]

# # When the user presses Enter
# if user_input:
#     relevant_article = find_most_relevant_article(user_input)
#     context = (
#         f"You are a knowledgeable assistant about Aluminium articles. "
#         f"Here is an article that may be relevant:\n"
#         f"Title: {relevant_article['title']}\n"
#         f"Description: {relevant_article['description']}\n"
#         f"Date: {relevant_article['date']}\n"
#         f"Link: {relevant_article['link']}\n"
#         "Please provide specific information or answer the user's question based on the article."
#     )
    
#     response = get_response(user_input, context)
#     st.write(f"Bot: {response}")
