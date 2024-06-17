"""Main module."""

import logging
from typing import Dict, Any
from typing import Optional

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from rich.logging import RichHandler

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

        answer = conversation.invoke(query)
        answer = answer["response"].strip()
        # answer = conversation.run(query)
        # answer = answer.strip()

        results = RAGResults(
            query=query,
            result=answer,
            source_documents=[""],
        )
        return results

    def run_chat(
        self,
        query: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        prompt_template=None,
        model_id="anthropic.claude-instant-v1",
    ) -> RAGResults:
        if not prompt_template:
            prompt_template = self.chat_prompt_template

        prompt = PromptTemplate(
            input_variables=["history", "query"], template=prompt_template
        )
        conversation = ConversationChain(
            prompt=prompt,
            llm=self.get_llm(model_id=model_id),
            verbose=True,
            memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
        )

        answer = conversation.invoke(query)
        answer = answer["response"].strip()
        # chat = ChatBedrock(model_id=model_id)
        # messages = [HumanMessage(content=query)]
        # response = chat(messages)
        # return RAGResults(query=query, result=response.content, source_documents=[""])
        return RAGResults(query=query, result=answer, source_documents=[""])
