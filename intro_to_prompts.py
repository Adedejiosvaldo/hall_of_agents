from dotenv import load_dotenv, find_dotenv
import os
import openai

load_dotenv(find_dotenv())
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
)


# Define the function to get completion
def get_completion(prompt: str, model=model):
    messages = [{"role": "user", "content": prompt}]
    # Use the model instance to get the completion
    # response = model.chat(messages=messages)
    # # Updated from openai.chat.completions
    response = model.invoke(messages)
    return response.content
 

# def get_completion(prompt: str, model=model):
#     messages = [{"role": "user", "content": prompt}]
#     response = model.chat
#     # response = openai.chat.completions(
#     #     model=model,
#     #     messages=messages,
#     # )
#     return response.choices[0].message["content"]


customer_review = """
Your product is terrible. You did
horribly. No one should ever buy this
from anyone. I want my money back.
"""

prompt = f"""
Rewrite the customer review in a more polite tone and then translate
the new review message into spanish
"""

rewrite = get_completion(prompt=prompt)

print(rewrite)
