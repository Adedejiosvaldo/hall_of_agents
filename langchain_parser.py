from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv, find_dotenv
import os
import openai

load_dotenv(find_dotenv())
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# model = AzureChatOpenAI(
#     azure_deployment="gpt-35-turbo",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2024-02-01",
#     temperature=0.7,
#     max_tokens=None,
#     timeout=None,
# )


from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GOOGLE_GENERATIVE_API_KEY"),
    # other params...
)

leave_time_schema = ResponseSchema(
    name="leave_time",
    description="when are they leaving for vacation to Europe. If there's an actual time written, use it, if not write unknown.",
)

leave_from_schema = ResponseSchema(
    name="leave_from",
    description="where are they leaving from, the airport or city name and state if available.",
)

cities_to_visit_schema = ResponseSchema(
    name="cities_to_visit",
    description="extract the cities they are going to visit. If there are more than one, put them in square brackets",
)

# response schema
response_schema = [leave_time_schema, leave_from_schema, cities_to_visit_schema]

# setup the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

format_instruction = output_parser.get_format_instructions()


email_template = """
From the following email, extract the following information:

leave_time: when are they leaving for vacation to Europe. If there's an actual
time written, use it, if not write unknown.

leave_from: where are they leaving from, the airport or city name and state if
available.

cities_to_visit: extract the cities they are going to visit.
If there are more than one, put them in square brackets like '["cityone", "citytwo"].

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email: {email}
{format_instruction}
"""

email_response = """
Here's our itinerary for our upcoming trip to Europe.
We leave from Denver, Colorado airport at 8:45 pm, and arrive in Amsterdam 10 hours later
at Schipol Airport.
We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before
taking a nap.

Some sightseeing will follow for a couple of hours.
We will then go shop for gifts
to bring back to our children and friends.

The next morning, at 7:45am we'll drive to to Belgium, Brussels - it should only take aroud 3 hours.
While in Brussels we want to explore the city to its fullest - no rock left unturned!

"""


prompt_template = ChatPromptTemplate.from_template(template=email_template)
messages_template = prompt_template.format_messages(
    email=email_response, format_instruction=format_instruction
)


response = model.invoke(messages_template)
print(response.content)
output_dict = output_parser.parse(response.content)
print(output_dict)
print(type(output_dict))
