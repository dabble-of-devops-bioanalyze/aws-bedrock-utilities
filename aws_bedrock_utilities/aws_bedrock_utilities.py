"""Main module."""

import os
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

# Keeping these here for backwards compatibility
from aws_bedrock_utilities.models.base import BedrockBase
from aws_bedrock_utilities.models.base import BedrockBase as BedrockUtils
