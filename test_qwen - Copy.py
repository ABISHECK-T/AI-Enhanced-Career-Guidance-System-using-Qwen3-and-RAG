from ollama import chat

response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "I want to become a Cloud Engineer"
        }
    ]
)

print(response["message"]["content"])