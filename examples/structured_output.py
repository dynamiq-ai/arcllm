"""
Structured output example for arcllm.

This example demonstrates JSON mode and JSON schema structured output.
"""

import json

import arcllm


def json_mode_example():
    """Example using JSON mode."""
    print("=== JSON Mode Example ===\n")

    response = arcllm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in JSON format.",
            },
            {
                "role": "user",
                "content": "List 3 programming languages with their year of creation.",
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    print(f"Raw response:\n{content}\n")

    # Parse the JSON
    data = json.loads(content)
    print(f"Parsed data: {json.dumps(data, indent=2)}")


def json_schema_example():
    """Example using JSON schema for strict structured output."""
    print("\n=== JSON Schema Example ===\n")

    # Define the expected schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
            "interests": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age", "email", "interests"],
    }

    response = arcllm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Generate a fake user profile for testing purposes.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "user_profile",
                "schema": schema,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    print(f"Raw response:\n{content}\n")

    # Parse and validate
    data = json.loads(content)
    print("Parsed profile:")
    print(f"  Name: {data['name']}")
    print(f"  Age: {data['age']}")
    print(f"  Email: {data['email']}")
    print(f"  Interests: {', '.join(data['interests'])}")


def extract_entities_example():
    """Example extracting structured entities from text."""
    print("\n=== Entity Extraction Example ===\n")

    text = """
    John Smith is a software engineer at TechCorp Inc. He has been working there
    since 2020 and specializes in machine learning. His office is located at
    123 Main Street, San Francisco, CA 94102. You can reach him at john.smith@techcorp.com.
    """

    schema = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name"],
            },
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                },
                "required": ["name"],
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["person", "company"],
    }

    response = arcllm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract structured information from the text.",
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entity_extraction",
                "schema": schema,
            },
        },
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    print("Extracted entities:")
    print(f"  Person: {data['person']['name']} - {data['person'].get('title', 'N/A')}")
    print(f"  Company: {data['company']['name']}")
    if "skills" in data:
        print(f"  Skills: {', '.join(data['skills'])}")


def main():
    """Run all structured output examples."""
    json_mode_example()
    json_schema_example()
    extract_entities_example()


if __name__ == "__main__":
    main()
