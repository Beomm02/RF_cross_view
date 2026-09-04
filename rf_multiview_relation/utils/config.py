from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if loaded is not None else {}
    except ModuleNotFoundError:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _strip_inline_comment(line: str) -> str:
    quote = ""
    for idx, char in enumerate(line):
        if char in ("'", '"'):
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
        elif char == "#" and not quote:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in _split_inline_list(inner)]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_inline_list(value: str) -> list[str]:
    parts = []
    start = 0
    quote = ""
    for idx, char in enumerate(value):
        if char in ("'", '"'):
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
        elif char == "," and not quote:
            parts.append(value[start:idx])
            start = idx + 1
    parts.append(value[start:])
    return parts
