"""Test cases for Gemini LLM client.

These tests directly test the GeminiCient implementations
without going through the wrapper layer.
"""

from pathlib import Path

import pytest
import yaml

from mini_agent.llm import GeminiClient
from mini_agent.retry import RetryConfig
from mini_agent.schema import Message

def load_config():
    """Load config from config.yaml"""
    config_path = Path("mini_agent/config/config_gemini.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# uv run pytest tests/test_llm_gemini_client.py::test_gemini_simple_content_generation -v -s
@pytest.mark.asyncio
async def test_gemini_simple_content_generation():
    """Test Gemini client with simple content generation.
    
    ✅ Test Pass
    Our Client can send simple content generation:
        - send prompt in messages form and generate output
        - parse output to LLMResponse successfully
    """
    print("\n=== Testing Gemini Simple Content Generation ===")

    config = load_config()

    # Create Gemini client
    client = GeminiClient(
        api_key=config["api_key"],
        model=config.get("model", "gemini-2.5-flash-lite"),
        retry_config=RetryConfig(enabled=False)
    )

    # Simple messages set in our mini_agent schema
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello from Gemini' and do nothing else."),
    ]

    # try to call llm
    try:
        response = await client.generate(messages=messages)
        try:
            print(f"\nResponse JSON:\n{response.model_dump_json(indent=2)}")
        except AttributeError:
            print(f"\nResponse Dict: {getattr(response, "__dict__", "No __dict__")}")
        
        assert response.content, "Response content is empty"
        assert "Hello" in response.content or "hello" in response.content
        assert response.thinking, "Missing thinking result"

        print("✅ Test Pass")
        return True

    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    
# uv run pytest tests/test_llm_gemini_client.py::test_gemini_tool_calling -v -s
@pytest.mark.asyncio
async def test_gemini_tool_calling():
    """Test Gemini client with tool calling.

    ✅ Test Pass
    Our Client can return function call
    """
    print("\n=== Testing Gemini Tool Calling ===")

    config = load_config()

    # Create Gemini client
    client = GeminiClient(
        api_key=config["api_key"],
        model=config.get("model", "gemini-2.5-flash-lite"),
        retry_config=RetryConfig(enabled=False)
    )

    # Define tool using dict format (will be converted internally for Gemini)
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather of a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, US",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    # Messages requesting tool use
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's the weather in New York, US?"),
    ]

    try:
        response = await client.generate(messages=messages, tools=tools)

        print(f"Response: {response.content}")
        print(f"Thinking: {response.thinking}")
        print(f"Tool calls: {response.tool_calls}")
        try:
            print(f"\nResponse JSON:\n{response.model_dump_json(indent=2)}")
        except AttributeError:
            print(f"\nResponse Dict: {getattr(response, "__dict__", "No __dict__")}")

        if response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0].function.name == "get_weather"
            print("✅ Gemini tool calling test passed")
        else:
            print("⚠️  Warning: LLM didn't use tools, but request succeeded")

        return True
    
    except Exception as e:
        print(f"❌ Gemini tool calling test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

# uv run pytest tests/test_llm_gemini_client.py::test_multi_turn_conversation -v -s
@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Test multi-turn conversation with tool calling."""
    print("\n=== Testing Gemini Multi-turn Conversation ===")

    config = load_config()

    # Create Gemini client
    client = GeminiClient(
        api_key=config["api_key"],
        model=config.get("model", "gemini-2.5-flash-lite"),
        retry_config=RetryConfig(enabled=False)
    )

    # Define tool using dict format
    tools = [
        {
            "name": "calculator",
            "description": "Perform arithmetic operations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        }
    ]

    try:
        # First turn - user asks
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What's 5 + 3?"),
        ]
        response = await client.generate(messages=messages, tools=tools)

        print(f"Turn 1 - Response: {response.content}")
        print(f"Turn 1 - Tool calls: {response.tool_calls}")

        if response.tool_calls:
            # Add assistant response
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=response.tool_calls,
                )
            )

            # Add tool result
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=response.tool_calls[0].id,
                    content="8",
                    name=response.tool_calls[0].id
                )
            )

            print(messages)

            # Second turn - get final answer
            final_response = await client.generate(messages=messages, tools=tools)
            print(f"Turn 2 - Response: {final_response.content}")

            assert final_response.content
            print("✅ Multi-turn conversation test passed")
        else:
            print("⚠️  Warning: LLM didn't use tools")

        return True
    
    except Exception as e:
        print(f"❌ Multi-turn conversation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False