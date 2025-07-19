# imports

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import gradio as gr
import json

load_dotenv(override=True)
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 


azureai = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version="2023-12-01-preview" 
)

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages)
    return response.choices[0].message.content

#gr.ChatInterface(fn=chat, type="messages").launch()

#Tools - can write a function and have the llm call htat funtion as part of its response

# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

#name
# description - tells AI when to call the function, and gives an example
# parameters

tools = [{"type": "function", "function": price_function}]

# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages, tools=tools) #passing in the tools

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages)
    
    return response.choices[0].message.content


gr.ChatInterface(fn=chat, type="messages").launch()