# GAIA Agent

Demonstration of an Agent-based system, which can be used for Evaluation
with [GAIA Benchmark](https://huggingface.co/papers/2311.12983).

This Agent doesn't use any Agent Framework, but is rather a plain Implementation using OpenAI API (AzureOpenAI to be
precise).
Thus the following env variables are required:

```bash
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_API_ENDPOINT=https://...
#Optional AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

The agent can be used in any python script:

```python
from gaia_agent import Agent

agent = Agent()
agent.answer_question("What is the capital of France?")
```
