from typing import Any


def list_to_string(array: list[Any]) -> str:
    if len(array) == 0:
        return ""

    if isinstance(array[0], list):
        return f"[{','.join([list_to_string(array_item) for array_item in array])}]"

    return f"[{','.join([str(element) for element in array])}]"
