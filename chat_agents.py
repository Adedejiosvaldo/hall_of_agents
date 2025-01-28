# from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
from langchain_openai import AzureChatOpenAI, AzureOpenAI

# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
# openai.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

model = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
)

llm = AzureOpenAI(
    azure_deployment="gpt-35-turbo",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    # temperature=0.7,
    # max_tokens=0,
    # timeout=0,
)
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)
# print(ai_msg.content)
print("________________")
print(llm.invoke("I love programming."))

# response = client.chat.completions.create(
#     model="gpt-35-turbo",  # model = "deployment_name".
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
#         {
#             "role": "assistant",
#             "content": "Yes, customer managed keys are supported by Azure OpenAI.",
#         },
#         {"role": "user", "content": "Do other Azure AI services support this too?"},
#     ],
# )

# print(response.choices[0].message.content)
