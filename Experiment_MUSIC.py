# Imports
import openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from IPython.display import Markdown, display

load_dotenv()  # This will load variables from the .env file

# Ensure environment variables are set
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Check for missing environment variables
if not endpoint or not api_key:
    raise EnvironmentError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY environment variables.")

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-07-01-preview",
    azure_endpoint=endpoint
)

completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[
        {"role": "user", "content": "Hello there"}
    ]
)

# Input the system prompt
system_prompt = """you are top notched AI music expert that have knowledge of all genres, songs, and artists. You need to google search lyrics. You have the following rules:\
1. Carefully break down what type of recommendation the user wants and the context.\
2. If asked to recommend genres similar to a song or artists please identify the top 3 genres.\
3. If asked to recommend artists from songs or genres then recommend the top 5 artists.
4. If asked to recommend songs from genres or artist than recommend the top 10 songs.
5. If asked for a general recommendation give them the top 5 songs based off of context.\
6. Be flexible and adaptable with recommendations and consider the context the user might ask.
7. always respond in markdown.
"""

# music recommender function
def music_recommender(user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )
    
    return response.choices[0].message.content

# User prompt (Change this to fit your needs!)
user_prompt = "Can you recommend me artists similar to Gorillaz"

# Example usage
response = music_recommender(user_prompt)
display(Markdown(response))

print(response)