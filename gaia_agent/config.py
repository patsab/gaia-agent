from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings, cli_enforce_required=True):
    """Configuration settings for the application.

    The Azure OpenAI Endpoint and API_KEY Vars need to be set in the environment.
    """

    AZURE_OPENAI_API_KEY: str = Field()
    AZURE_OPENAI_API_ENDPOINT: str = Field()
    AZURE_OPENAI_API_VERSION: str = "2025-01-01-preview"

    AGENT_MODEL_NAME: str = "gpt-4.1-mini"
    AGENT_REASONING_MODEL_NAME: str = "o4-mini"


CONFIG = Config()
