# imports
import gradio as gr
import json
from gemini_client import generate_city_image, generate_image
from openai_client import chat, chat_img

def run_gradio():
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500, label="FlightAI Assistant", show_copy_button=True, type="messages") #added the type function so gradio knows to expect messages
            # added the label function for UI purposes, and the show copy button allows the user to copy the content of the messages
            image_output = gr.Image(height=500, label="Generated Image")
        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:") 
        with gr.Row():
            clear = gr.Button("Clear") #clear button

        def do_entry(message, history):
            history = history or []
            history.append({"role": "user", "content": message})
            return "", history

        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            chat, inputs=chatbot, outputs=[chatbot, image_output]
        )

        clear.click(lambda: ([], None), inputs=None, outputs=[chatbot, image_output], queue=False)

    ui.launch(inbrowser=True)

def main():
    history = [{"role": "user", "content": "How much is a ticket to Paris?"}]
    history, image = chat(history)
    print(history[-1]["content"])
    if image:
        image.show()

run_gradio()