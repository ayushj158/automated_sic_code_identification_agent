import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
# from langchain_xai import ChatXAI

load_dotenv()
llm = ChatOllama(model="llama3.1:8b")
# llm = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#     api_key=os.environ["AZURE_OPENAI_API_KEY"]
# )


# llm = ChatXAI(
#     xai_api_key=os.environ["GROK_API_KEY"],
#     model="grok-beta",
# )
