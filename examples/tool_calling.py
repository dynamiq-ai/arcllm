"""
Tool calling example for arcllm.

This example demonstrates function/tool calling with the unified API.
"""

import json

import arcllm

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather function."""
    return json.dumps(
        {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "conditions": "sunny",
        }
    )


def search_web(query: str) -> str:
    """Mock search function."""
    return json.dumps(
        {
            "query": query,
            "results": [{"title": "Example result", "snippet": f"Information about {query}"}],
        }
    )


def execute_tool(tool_call: arcllm.ToolCall) -> str:
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = tool_call.function.parse_arguments()

    if name == "get_weather":
        return get_weather(**args)
    if name == "search_web":
        return search_web(**args)
    return json.dumps({"error": f"Unknown function: {name}"})


def main():
    """Run tool calling example."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to weather and search tools.",
        },
        {"role": "user", "content": "What's the weather like in San Francisco?"},
    ]

    print("User: What's the weather like in San Francisco?\n")

    # First call - model decides to use tools
    response = arcllm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let model decide
    )

    assistant_message = response.choices[0].message
    print(f"Assistant response: {assistant_message.content}")

    # Check if model wants to call tools
    if assistant_message.tool_calls:
        print(f"\nModel wants to call {len(assistant_message.tool_calls)} tool(s):")

        # Add assistant message to conversation
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls],
            }
        )

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            print(f"\n  Tool: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

            # Execute the tool
            result = execute_tool(tool_call)
            print(f"  Result: {result}")

            # Add tool result to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Second call - model processes tool results
        print("\n--- Getting final response ---\n")
        final_response = arcllm.completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        print(f"Final response: {final_response.choices[0].message.content}")
    else:
        print("Model did not use any tools.")


def streaming_tool_calls():
    """Example of streaming with tool calls."""
    print("\n--- Streaming with tool calls ---\n")

    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
    ]

    stream = arcllm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        stream=True,
    )

    # Collect chunks
    chunks = []
    for chunk in stream:
        chunks.append(chunk)

        # Stream content if present
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

        # Check for tool calls in delta
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
                if tc.get("function", {}).get("name"):
                    print(f"\n[Tool call: {tc['function']['name']}]", end="")
                if tc.get("function", {}).get("arguments"):
                    print(tc["function"]["arguments"], end="")

    print()

    # Build complete response
    response = arcllm.stream_chunk_builder(chunks)
    if response.choices[0].message.tool_calls:
        print(f"\nComplete tool calls: {len(response.choices[0].message.tool_calls)}")
        for tc in response.choices[0].message.tool_calls:
            print(f"  - {tc.function.name}: {tc.function.arguments}")


if __name__ == "__main__":
    main()
    # streaming_tool_calls()
