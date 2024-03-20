"""Main module."""

import boto3
import pprint
from typing import Dict, Any, TypedDict, List
from langchain_core.documents.base import Document
from botocore.client import Config
from langchain.llms.bedrock import Bedrock
from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
import os
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

bedrock_config = Config(
    connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
)
bedrock_client = boto3.client("bedrock-runtime")
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config)

model_kwargs = {
    "claude": {"temperature": 0, "top_k": 10, "max_tokens_to_sample": 3000},
    "llama2": {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_gen_len": 512,
    },
}


def get_model_ids():
    client = boto3.client("bedrock")
    models = client.list_foundation_models()
    model_ids = [model["modelId"] for model in models["modelSummaries"]]
    return model_ids


class RAGResults(TypedDict):
    query: str
    result: str
    source_documents: List[Document | str]


class BedrockUtils:
    def __init__(self, connect_timeout: int = 120, read_timeout: int = 120):
        self.bedrock_config = Config(
            connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
        )
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.bedrock_agent_client = boto3.client(
            "bedrock-agent-runtime", config=bedrock_config
        )
        self.models = [
            "anthropic.claude-instant-v1",
            "anthropic.claude-v2",
            "meta.llama2-13b-chat-v1",
            "meta.llama2-70b-chat-v1",
        ]
        self.model_kwargs = {
            "claude": {
                "temperature": 0, "top_k": 10,
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
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""
        self.chat_prompt_template = """
Human: You are an AI system for knowledge retrieval. You provide answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in {input} tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Current conversation:
{history}

The response should be specific and use statistics or numbers when possible.
User: {input}
Bot:
    """

    def get_models_args(self, model: str):
        for key in self.model_kwargs.keys():
            if key in model:
                logging.info(f"Getting models args for {key}")
                return self.model_kwargs[key]
        logging.warning(f"Model args not found for {model}")
        return {}

    def get_llm(self, model_id: str):
        args = self.get_models_args(model_id)
        llm = BedrockChat(
            model_id=model_id,
            model_kwargs=args,
            # client=bedrock_client,
        )
        return llm

    def get_retriever(self, knowledge_base_id) -> AmazonKnowledgeBasesRetriever:
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
            # region_name=AWS_REGION,
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
            template=prompt_template, input_variables=["context", "question"]
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
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.chat_prompt_template

        prompt = PromptTemplate(
            input_variables=["history", "input"], template=prompt_template
        )
        memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
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
        messages = [
            HumanMessage(
                content=query
            )
        ]
        response = chat(messages)
        return RAGResults(
            query=query,
            result=response.content,
            source_documents=[""]
        )
