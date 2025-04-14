#!/usr/bin/env python

"""Tests for `aws_bedrock_utilities` package."""

import pytest

from click.testing import CliRunner

from aws_bedrock_utilities import aws_bedrock_utilities
from aws_bedrock_utilities import cli
import os
from pprint import pprint

from aws_bedrock_utilities.models.bedrock_chat import BedrockChatWrapper
from aws_bedrock_utilities.models.bedrock_knowledgebase import (
    BedrockKnowledgeBaseChatWrapper,
)


# these tests can only be run locally since they require aws credentials


def test_chat():
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    chat = BedrockChatWrapper()
    response = chat.run_chat(
        query="What is the capital of France?", model_id="anthropic.claude-instant-v1"
    )
    pprint(response)
    assert True


def test_chat_with_memory():
    chat = BedrockChatWrapper()
    response = chat.run_chat_with_memory(
        query="What is the capital of France?", model_id="anthropic.claude-instant-v1"
    )
    pprint(response)
    assert True


def test_chat_with_rag():
    rag_chat = BedrockKnowledgeBaseChatWrapper()
    response = rag_chat.run_kb_chat(
        query="Tell me something interesting about this dataset.",
        knowledge_base_id=os.environ["KB_ID"],
    )
    pprint(response)
    assert True
