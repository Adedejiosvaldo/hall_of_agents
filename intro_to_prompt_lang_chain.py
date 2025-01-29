from dotenv import load_dotenv, find_dotenv
import os
import openai

load_dotenv(find_dotenv())
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

model = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
)


customer_review = ""
template_string = f"""
    Rewrite {customer_review} in a more polite tone and then translate
    the new review message into spanish
    """


# Define the function to get completion
def get_completion():
    prompt_template = ChatPromptTemplate.from_template(template_string)
    translation_message = prompt_template.format_messages(
        customer_review=customer_review
    )
    response = model.invoke(translation_message)
    return response.content


# prompt_template = ChatPromptTemplate.from_template(template_string)
# translation_message = prompt_template.format_messages(customer_review=customer_review)
# response = model(translation_message)

if __name__ == "__main__":
    customer_review = input("Enter the review: ")

    print("_____________Loading_____________")
    # Get the completion
    print(get_completion())
    # print(rewrite)
