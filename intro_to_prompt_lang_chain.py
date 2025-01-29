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


# customer_review = ""
# company_name = ""
# product_name = ""
# template_string = f"""
#     Rewrite {customer_review} in a more polite tone and then translate
#     the new review message into spanish. The company name is {company_name}
#     and make sure to mention the product name {product_name}
#     """


# Define the function to get completion
def get_completion(customer_review: str, company_name: str, product_name: str):
    template_string = f"""
    Please rewrite the following review in a more polite and professional tone, making sure to mention the company name {company_name} and the product name {product_name}:
    "{customer_review}"
    Then, translate the revised review into Spanish.
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    translation_message = prompt_template.format_messages(
        customer_review=customer_review,
        company_name=company_name,
        product_name=product_name,
    )
    response = model.invoke(translation_message)
    return response.content


# prompt_template = ChatPromptTemplate.from_template(template_string)
# translation_message = prompt_template.format_messages(customer_review=customer_review)
# response = model(translation_message)

if __name__ == "__main__":
    customer_review = input("Enter the review: ")
    company_name = input("Enter the company name: ")
    product_name = input("Enter the product name: ")
    print("_____________Loading_____________")
    # Get the completion
    print(print(get_completion(customer_review, company_name, product_name)))
    # print(rewrite)
