from typing import Any


def remove_schema_fields(obj: Any) -> Any:
    """MCP tools often have $schema fields that cause issues with Google Gemini API.
    This function recursively removes any $schema fields from a nested dictionary or list."""
    if isinstance(obj, dict):
        return {k: remove_schema_fields(v) 
                for k, v in obj.items() 
                if k != '$schema'}
    elif isinstance(obj, list):
        return [remove_schema_fields(item) for item in obj]
    else:
        return obj