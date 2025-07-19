import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from gemini_client import generate_image
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

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

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

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

image_gen_func = {
    "name": "get_image",
    "description": "Generate an image based on the city of choice. Call this whenever you need to show a picture of a city",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A description of the image to generate"
            },
            "style": {
                "type": "string",
                "description": "An optional stylistic choice, like 'photorealistic', 'surrealism', or 'popart'"
            },
        },
        "required":  ["prompt"],
        "additionalProperties": False
    }
}

tools = [ 
    { 
        "type": "function",
        "function": price_function
    },
    {
        "type": "function",
        "function": image_gen_func
    }
]

def chat_img(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages)
    return response.choices[0].message.content


def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

def get_image(prompt):
    print(f"Tool get_image called for {prompt}")
    image = generate_image(prompt)
    
    if image is None:
        return None  # Or raise an error

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64


def handle_tool_call(message):
    responses = []
    image = None

    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)

        if tool_call.function.name == 'get_ticket_price':
            city = arguments.get('destination_city')
            price = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": json.dumps({"destination_city": city, "price": price}),
                "tool_call_id": tool_call.id
            })

        elif tool_call.function.name == 'get_image':
            prompt = arguments.get('prompt')
            image = generate_image(prompt)

            responses.append({
                "role": "tool",
                "content": json.dumps({
                    "prompt": prompt
                }),
                "tool_call_id": tool_call.id
            })

    return responses, image


def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages, tools=tools) #passing in the tools
    image = None

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, image = handle_tool_call(message)
        messages.append(message)

        for res in response:
            messages.append(res)

        response = azureai.chat.completions.create(model='gpt-4o-mini', messages=messages)

    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]
    
    return history, image