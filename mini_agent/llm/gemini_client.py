"""Gemini LLM client implementation."""

import logging

from typing import Any

from google import genai
from google.genai import types

from .base import LLMClientBase
from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)

class GeminiClient(LLMClientBase):
    """LLM client using Gemini's protocol.

    This client uses the official Gemini SDK and supports:
    - Reasoning content (via reasoning_split=True) # Not implemented
    - Tool calling # Not implemented
    - Retry logic # Not implemented
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            model: Model name to use (default: "gemini-2.5-flash-lite")
            retry_config: Optional retry configuration
        """
        super().__init__(
            api_key=api_key,
            api_base="google", # no use in gemini case, but we pass in a mock one anyway
            model=model, 
            retry_config=retry_config
        )

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)

    def _convert_messages(
        self, 
        messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Gemini format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_instruction, contents)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            # For system instruction
            if msg.role == "system":
                system_instruction=msg.content
                continue

            # For user message:
            if msg.role == "user":
                contents.append(
                    types.UserContent(
                        parts=[
                            types.Part(
                                text=msg.content,
                                thought_signature=msg.thought_signature
                                )
                        ]
                    )
                )

            # For assistant("model" in google api) messages
            elif msg.role == "assistant":
                assistant_part = []
                # Add content if present
                if msg.content:
                    assistant_part.append(types.Part.from_text(text=msg.content))
                # Add extended thinking content if present
                if msg.thinking:
                    assistant_part.append(
                        types.Part(
                            thought=True,
                            text=msg.thinking,
                        )
                    )
                # Add tool calls if present
                if msg.tool_calls:
                    tool_parts = []
                    for i, tool_call in enumerate(msg.tool_calls):
                        tool_parts.append(
                            types.Part(
                                thought_signature=msg.thought_signature if i == 0 and msg.thought_signature is not None else None,
                                function_call=types.FunctionCall(
                                    name=tool_call.function.name,
                                    args=tool_call.function.arguments
                                )
                            )
                        )
                    assistant_part.extend(tool_parts)

                contents.append(types.ModelContent(parts=assistant_part))
                
            # For tool result messages
            elif msg.role == "tool":
                # msg.content can be str or list[dict[str, Any]]
                # gemini format response need dict format
                # untested code
                if type(msg.content) is str:
                    part = types.Part.from_function_response(
                        name=msg.name,
                        response={"result":msg.content},
                    )
                    contents.append(
                        types.UserContent(
                            parts=[part],
                        )
                    )
                else:
                    part = types.Part.from_function_response(
                        name=msg.name,
                        response=msg.content,
                    )
                    contents.append(
                        types.UserContent(
                            parts=[part]
                        )
                    )


        # if system instruction not shown in messages
        if system_instruction is None:
            system_instruction = "You are a helpful assistant."

        return system_instruction, contents

    def _convert_tools(self, tools: list[Any] | None) -> list[types.Tool]:
        """Convert tools to Gemini format.

        Args:
            tools: List of Tool objects or dicts, or None

        Returns:
            List of tools in Gemini types.Tool
        """
        if tools is None:
            return []

        tool_list = []
        
        # tool is dict[str, Any]
        # detail: https://googleapis.github.io/python-genai/index.html#manually-declare-and-invoke-a-function-for-function-calling
        for tool in tools:
            if isinstance(tool, dict):
                # if already a dict, check if it's in OpenAI format
                if "type" in tool and tool["type"] == "function":
                    function = tool.get("function", {})
                    tool_list.append(
                        types.FunctionDeclaration(
                                name=function["name"],
                                description=function["description"],
                                parameters=function["parameters"],
                        )
                    )
                else:
                    # Assume it's in Anthropic format, convert to Google schema
                    tool_list.append(
                        types.FunctionDeclaration(
                                name=tool["name"],
                                description=tool["description"],
                                parameters=tool["input_schema"],
                            )
                        )
            elif hasattr(tool, "to_google_schema"):
                tool_list.append(tool.to_google_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")

        return [types.Tool(function_declarations=tool_list)] if len(tool_list) > 0 else []
    
    def _parse_response(self, response: types.GenerateContentResponse) -> LLMResponse:
        """Parse Gemini response into LLMResponse.

        Args:
            response: Gemini generate_content response

        Returns:
            LLMResponse object
        """
        # Gemini may return multiple response candidate, we choose first one as result
        # Candidate includes all kinds of results, like thinkng, tools, code, text generation
        candidate = response.candidates[0]
        response_content = ""
        thinking = None
        tool_calls = None
        thought_signature = None

        for i, part in enumerate(candidate.content.parts):
            if part.thought:
                # Get thinking content
                thinking = part.text
            elif part.function_call:
                # Get tool call message
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=part.function_call.name,
                        type="function",
                        function=FunctionCall(
                            name=part.function_call.name,
                            arguments=part.function_call.args
                        ),
                    )
                )
            else:
                # Get text generate content
                response_content = part.text

            if part.thought_signature is not None:
                thought_signature = part.thought_signature

        # Get finish_reason
        finish_reason = candidate.finish_reason

        # Get token usage if possible
        usage = TokenUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            total_tokens=response.usage_metadata.total_token_count,
        )

        # Return results
        return LLMResponse(
            content=response_content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thought_signature=thought_signature
        )


    def _prepare_request(
        self, 
        messages: list[Message], 
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for Gemini API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
                system_instruction: the system instruction passed into config later
                api_messages: gemini format messages
                tools: tools informations
        """
        system_message, api_messages = self._convert_messages(messages)
        tools = self._convert_tools(tools)

        return {
            "system_instruction": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }

    async def _make_api_request(
            self,
            system_instruction: str,
            api_messages: list[dict[str, Any]],
            tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentResponse:
        """Execute API request (core method that can be retried).

        Args:
            system_instruction: System instruction pass to config
            api_messages: List of messages in Gemini format
            tools: Optional list of tools

        Returns:
            Gemini ChatCompletion response (full response including usage)
        Raises:
            Exception: API call failed
        """
        config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                # TODO: hardcode thinking config for now, maybe later we make it configurable
                thinking_config= {
                    "include_thoughts": True,
                    "thinking_budget": -1,
                    # You can only set only one of thinking budget and thinking level.
                    # "thinking_level": "MEDIUM",
                },
            )
        if tools:
            config.tools = tools
            config.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='AUTO'))
            

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=api_messages,
            config= config,
        )

        return response
    
    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from Gemini LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse containing the generated content
        """
        # Prepare request
        request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["system_instruction"],
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["system_instruction"],
                request_params["api_messages"],
                request_params["tools"],
            )

        # Parse and return response
        return self._parse_response(response)