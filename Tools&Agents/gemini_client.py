#image generation with google api
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
# Some imports for handling images

import base64
from io import BytesIO
from PIL import Image

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

def gemini_run_image():
    user_prompt = input("Enter a description of the image you want to generate: ")
    messages = user_prompt

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=[{"role": "user", "parts": [user_prompt]}],
        config=types.GenerateContentConfig(
        response_modalities = ['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save('gemini-native-image.png')
            #image.show()

#Implementing an Agent Framework
def generate_city_image(city_description):
    contents = f"Generate a photo-realistic image of {city_description} from a travel brochure."
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    )

    image = None
    for part in response.parts:
        if hasattr(part, "inline_data"):
            image_data = base64.b64decode(part.inline_data.data)
            image = Image.open(BytesIO(image_data))
            image.save("city_image.png")
            image.show()
        elif hasattr(part, "text"):
            print("Text Output:", part.text)
    return image

def chat(history):
    image = None

    messages = [{"role": "user", "parts": [item["content"]]} for item in history if item["role"] == "user"]

    response = client.generate_content(
        contents=messages,
        generation_config=types.GenerationConfig(response_mime_types=["text/plain", "image/png"])
    )

    reply_text = ""
    for part in response.parts:
        if hasattr(part, "text") and part.text:
            reply_text += part.text
        elif hasattr(part, "inline_data"):
            image_data = base64.b64decode(part.inline_data.data)
            image = Image.open(BytesIO(image_data))
            image.save("generated_response.png")

    history.append({"role": "assistant", "content": reply_text.strip()})
    return history, image

def generate_image(prompt):
    if prompt is None or not isinstance(prompt, str) or prompt.strip() == "":
        raise ValueError("Prompt must be a non-empty string")
    
    print(f"Prompt received: '{prompt}'")

    contents = prompt

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    )

    image = None
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                print("Gemini:", part.text)
            elif hasattr(part, "inline_data") and part.inline_data:
                image = Image.open(BytesIO(part.inline_data.data))
                return image
    else:
        print("Empty response or no candidates returned.")

    return image