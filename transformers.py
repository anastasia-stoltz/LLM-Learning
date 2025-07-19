from ollama import chat

response = chat("codeqwen", [
            {
                "role": "user",
                "content": "Generate C++ code that prints out hello world"
            }
            ])

"gemma"

print(response.message.content)