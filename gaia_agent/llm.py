from openai import AzureOpenAI

from .config import CONFIG


def get_llm() -> AzureOpenAI:
    """Return Azure OpenAI Client Instance."""
    return AzureOpenAI(
        api_version=CONFIG.AZURE_OPENAI_API_VERSION,
        api_key=CONFIG.AZURE_OPENAI_API_KEY,
        azure_endpoint=CONFIG.AZURE_OPENAI_API_ENDPOINT,
    )
