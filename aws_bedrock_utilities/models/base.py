"""Main module."""

import boto3
import pprint
import json
import hashlib
import funcy
from typing import Dict, Any, TypedDict, List, Optional
from langchain_core.documents.base import Document
from botocore.client import Config
from langchain.llms.bedrock import Bedrock
from langchain.retrievers.bedrock import (
    AmazonKnowledgeBasesRetriever,
    RetrievalConfig,
    VectorSearchConfig,
)
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.embeddings import (
    BedrockEmbeddings,
)  # to create embeddings for the documents.
from langchain_experimental.text_splitter import (
    SemanticChunker,
)  # to split documents into smaller chunks.
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

import os
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class RAGResults(TypedDict):
    query: str
    result: str
    source_documents: List[Document | str]


class BedrockBase:
    def __init__(
        self,
        connect_timeout: int = 120,
        read_timeout: int = 120,
        bedrock_client: Optional = None,
        bedrock_runtime_client: Optional = None,
        bedrock_agent_client: Optional = None,
    ):

        self.bedrock_config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": 0},
        )
        if not bedrock_client:
            self.bedrock_client = boto3.client("bedrock")
        else:
            self.bedrock_client = bedrock_client
        if not bedrock_runtime_client:
            self.bedrock_runtime_client = boto3.client("bedrock-runtime")
        else:
            self.bedrock_runtime_client = bedrock_runtime_client
        if not bedrock_agent_client:
            self.bedrock_agent_client = boto3.client(
                "bedrock-agent-runtime", config=self.bedrock_config
            )
        else:
            self.bedrock_agent_client = bedrock_agent_client
        self.models = [
            "anthropic.claude-instant-v1",
            "anthropic.claude-v2",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "meta.llama2-13b-chat-v1",
            "meta.llama2-70b-chat-v1",
        ]
        self.model_kwargs = {
            "claude": {
                "temperature": 0,
                "top_k": 10,
                # in claudev3 this may be max_tokens
                # "max_tokens_to_sample": 3000
            },
            "llama2": {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_gen_len": 512,
            },
        }
        self.kb_prompt_template = """
    Human: You are an AI system for knowledge retrieval. You provide answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <input> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    <context>
    {context}
    </context>

    <input>
    {input}
    </input>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""
        self.chat_prompt_template = """
Human: You are an AI system for knowledge retrieval. You provide answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in {query} tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Current conversation:
{history}

The response should be specific and use statistics or numbers when possible.
User: {query}
Bot:
    """

    def get_model_ids(self, client: Optional = None):
        if not client:
            client = boto3.client("bedrock")
        models = client.list_foundation_models()
        model_ids = [model["modelId"] for model in models["modelSummaries"]]
        return model_ids

    def get_models_args(self, model: str):
        for key in self.model_kwargs.keys():
            if key in model:
                # logging.info(f"Getting models args for {key}")
                return self.model_kwargs[key]
        logging.warning(f"Model args not found for {model}")
        return {}

    def get_llm(self, model_id: str):
        args = self.get_models_args(model_id)
        llm = ChatBedrock(
            model_id=model_id,
            model_kwargs=args,
            # client=bedrock_client,
        )
        return llm

    def get_retriever(
        self, knowledge_base_id: str, n_results: int = 4
    ) -> AmazonKnowledgeBasesRetriever:
        retrieval_config = RetrievalConfig(
            vectorSearchConfiguration=VectorSearchConfig(numberOfResults=n_results)
        )

        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config=retrieval_config,
        )
        return retriever

    def run_kb_chat(
        self,
        query: str,
        knowledge_base_id: str,
        prompt_template=None,
        model_id="anthropic.claude-instant-v1",
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.kb_prompt_template
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.get_llm(model_id=model_id),
            chain_type="stuff",
            retriever=self.get_retriever(knowledge_base_id=knowledge_base_id),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        answer = qa(query)
        return answer

    def run_chat_with_memory(
        self,
        query: str,
        prompt_template=None,
        model_id="anthropic.claude-instant-v1",
        memory: ConversationBufferMemory = None,
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.chat_prompt_template
        if not memory:
            memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")

        prompt = PromptTemplate(
            input_variables=["history", "query"], template=prompt_template
        )
        conversation = ConversationChain(
            prompt=prompt,
            llm=self.get_llm(model_id=model_id),
            verbose=True,
            memory=memory,
        )

        answer = conversation.run(query)
        answer = answer.strip()

        results = RAGResults(
            query=query,
            result=answer,
            source_documents=[""],
        )
        return results

    def run_chat(
        self,
        query: str,
        model_kwargs: Dict[str, Any],
        prompt_template=None,
        model_id="anthropic.claude-instant-v1",
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.chat_prompt_template

        chat = BedrockChat(
            model_id=model_id,
        )
        messages = [HumanMessage(content=query)]
        response = chat(messages)
        return RAGResults(query=query, result=response.content, source_documents=[""])
