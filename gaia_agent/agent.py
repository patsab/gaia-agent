import json
from typing import Callable, Literal

from openai import AzureOpenAI
from openai.types import FunctionDefinition
from openai.types.chat.chat_completion import Choice

from .config import CONFIG
from .llm import get_llm
from .tools import get_tool_list

SYSTEM_PROMPT = """You are an AI assistant, who is responsable for answering the user question.
You can use the available tools to answer the question."""


class Agent:
    """GAIA Agent Class."""

    llm: AzureOpenAI
    tools: dict[str, tuple[Callable, FunctionDefinition]]
    tool_definitions: list[FunctionDefinition]
    model_name: str = CONFIG.AGENT_MODEL_NAME
    reasoning_model_name: str = CONFIG.AGENT_REASONING_MODEL_NAME
    reasoning_effort: Literal["low", "medium", "high"] = "low"

    def __init__(self):
        self.llm = get_llm()
        self.tools = get_tool_list()
        self.tool_definitions = [tool[1] for tool in self.tools.values()]

    def answer_question(self, question: str) -> str:
        """Answer a question using the LLM and the available tools."""
        # Init question answering
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        task_finished = False

        # agent loop
        while not task_finished:
            # Call the LLM with the current messages
            response = self.llm_call(messages)
            # if finish_reason is "stop", we are done
            if response.finish_reason == "stop":
                messages.append({"role": "assistant", "content": response.message.content})
                task_finished = True
            elif response.message.tool_calls is not None:
                messages = self.call_tool_and_append_result(llm_response=response, messages=messages)
            else:
                return f"Error: Unknown stop reason: {response.finish_reason}"

        # final formatting
        return self.format_response(messages[-1]["content"])

    def call_tool_and_append_result(self, llm_response: Choice, messages: list) -> list:
        """Call the requested tool and return the result as a string."""
        messages.append(llm_response.message)
        for tool_call in llm_response.message.tool_calls:
            # args are returned as a string, so we need to parse them to a dict
            arguments = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )
            # call the tool and append the result
            tool_result = self.tools[tool_call.function.name][0](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(json.dumps(tool_result)),
                },
            )
        return messages

    def llm_call(self, messages: list, use_reasoning: bool = False) -> Choice:
        """Call the LLM with the given messages and return the response."""
        gpt_request_params = {
            "model": self.reasoning_model_name if use_reasoning else self.model_name,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in self.tool_definitions],
        }
        if use_reasoning:
            gpt_request_params["reasoning_effort"] = self.reasoning_effort

        res = self.llm.chat.completions.create(**gpt_request_params)
        return res.choices[0]

    def format_response(self, response: str) -> str:
        """Format the response from the LLM."""
        format_prompt = (
            "You are an AI assistant, who is responsable for formatting the response of the LLM. "
            "You get the answer from another Agent and you just need to format it. "
            "The Answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "  # noqa: E501
            "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "  # noqa: E501
            "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise."  # noqa: E501
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."  # noqa: E501
        )
        messages = [{"role": "system", "content": format_prompt}, {"role": "user", "content": response}]
        return self.llm.chat.completions.create(model=self.model_name, messages=messages).choices[0].message.content
