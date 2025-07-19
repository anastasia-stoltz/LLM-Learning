# imports

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import gradio as gr

load_dotenv(override=True)
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 


azureai = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version="2023-12-01-preview" 
)

system_message = "You are a helpful assistant"

#We will write a function chat(message, history) where:
#message is the prompt to use
#history is the past conversation, in OpenAI format

#We will combine the system message, history and latest message, then call OpenAI.

def chat(message, history): #function for conversational ai
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}] 

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    stream = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages, stream=True)

    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            yield response
    # changing the code so that it ensures chunk.choices is not empty and that delta content exists


#turn this into a user interface with an instant method style of interaction
#gr.ChatInterface(fn=chat, type="messages").launch()


#practicing multishot prompting and context enrichment

system_message = "You are a helpful assistant at a 4-year college. You should try to recommend some graduate programs \
    for an AI course. For example, if the students says 'I'm looking for international graduate programs' you could reply with \
        international colleges that has AI programs for graduate students. Encourage the student to find a unique experience in their graduate degree."


def chat(message, history): #function for conversational ai
   
    relevant_system_message = system_message
    if 'undergraduate' in message: #adding this to predefine a response to a certain phrase - could be used to prevent harmful use of AI?
        # could have it be a dictionary to beef it up a little and make it more complex
        #RAG subsitute ?
        relevant_system_message += "Unfortunately, I only specialize in graduate programs"
   
   
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}] 

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    stream = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages, stream=True)

    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            yield response

gr.ChatInterface(fn=chat, type="messages").launch()