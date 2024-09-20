import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = getpass.getpass()


# Model from the langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model.invoke(messages)
    # model.invoke exit: 
    # AIMessage(content='ciao!', response_metadata={'token_usage': 
    #   {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23}, 'model_name': 
    #   'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None},
    #   id='run-fc5d7c88-9615-48ab-a3c7-425232b562c5-0')


# Output Parser from the langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = model.invoke(messages)

parser.invoke(result)

chain = model | parser

chain.invoke(messages)


# Prompt Template from the langchain_core.prompts import ChatPromptTemplate


system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "italian", "text": "hi"})

result
