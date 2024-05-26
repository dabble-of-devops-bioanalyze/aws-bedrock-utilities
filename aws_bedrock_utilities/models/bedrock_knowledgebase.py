"""Main module."""

import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers.bedrock import (
    AmazonKnowledgeBasesRetriever,
    RetrievalConfig,
    VectorSearchConfig,
)
from rich.logging import RichHandler

from aws_bedrock_utilities.models.base import BedrockBase, RAGResults

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class BedrockKnowledgeBaseChatWrapper(BedrockBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "prompt_template" not in kwargs:
            self.prompt_template = self.kb_prompt_template

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
        retriever = self.get_retriever(knowledge_base_id=knowledge_base_id)

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.get_llm(model_id=model_id),
            prompt=prompt,
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        response = retrieval_chain.invoke(
            {
                "input": query,
            }
        )
        source_documents = response["context"]
        answer = response["answer"]
        query = query
        answer = RAGResults(
            source_documents=source_documents,
            result=answer,
            query=query,
        )
        return answer
