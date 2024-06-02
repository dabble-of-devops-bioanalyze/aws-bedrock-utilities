"""Top-level package for AWS Bedrock Utilities."""

__author__ = """Jillian Rowe"""
__email__ = "jillian@dabbleofdevops.com"
__version__ = "0.1.0"

from aws_bedrock_utilities.models import (
    BedrockBase,
    BedrockChatWrapper,
    BedrockKnowledgeBaseChatWrapper,
)

DEFAULT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
