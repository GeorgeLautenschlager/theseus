from __future__ import annotations

import json
import re


def parse_json_response(raw_response: str) -> dict:
    """Parses a model's JSON reply, tolerating ```json ... ``` code fences."""
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    json_str = match.group(0) if match else raw_response
    return json.loads(json_str)
