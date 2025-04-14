"""Main module."""

import logging
from typing import Dict, Any, List, Optional

from rich.logging import RichHandler

from langchain_core.prompts import ChatPromptTemplate
from aws_bedrock_utilities.models.base import BedrockBase, RAGResults

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class BedrockChatWrapper(BedrockBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "prompt_template" not in kwargs:
            self.prompt_template = self.chat_prompt_template

    def run_chat_with_memory(
        self,
        query: str,
        prompt_template: Optional[ChatPromptTemplate] = None,
        model_id="anthropic.claude-instant-v1",
        history: Optional[str] = None,
    ) -> RAGResults:
        message_history = ""
        if not prompt_template:
            prompt_template = self.chat_prompt_template
        if history and len(history):
            message_history = f"""Here's the message history.
            <history>
            {history}
            </history>
            """

        query = f"""
        {message_history}
        <query>
        {query}
        </query>
        """

        prompt = prompt_template
        chain = prompt | self.get_llm(model_id=model_id)
        answer = chain.invoke(
            {
                "input": query,
            }
        )
        response = answer.content.strip()
        return RAGResults(query=query, result=response, source_documents=[""])

    def run_chat(
        self,
        query: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[ChatPromptTemplate] = None,
        model_id: str = "anthropic.claude-instant-v1",
    ) -> RAGResults:
        if not prompt_template:
            prompt = self.chat_prompt_template
        else:
            prompt = prompt_template

        chain = prompt | self.get_llm(model_id=model_id)
        answer = chain.invoke(
            {
                "input": query,
            }
        )
        response = answer.content.strip()
        return RAGResults(query=query, result=response, source_documents=[""])
