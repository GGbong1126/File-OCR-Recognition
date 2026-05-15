from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FieldSpec:
    """Field spec passed from frontend/API.

    key: output json key, e.g. "name"
    display_name: human readable name shown to the model, e.g. "姓名"
    description: optional extra instruction, e.g. "统一输出YYYY-MM" or enum hints
    """

    key: str
    display_name: str
    description: str | None = None


_SYSTEM_PROMPT = (
    "你是一个信息抽取引擎。\n"
    "你的任务：从给定文本中抽取指定字段，并严格输出JSON对象。\n"
    "硬性要求：\n"
    "1) 只输出JSON（不要代码块、不要解释、不要多余文字）。\n"
    "2) JSON必须是一个对象，键只能来自我给你的key列表。\n"
    "3) 如果文本中没有明确证据，值返回空字符串\"\"，不要猜测/编造。\n"
    "4) 值统一输出为字符串。\n"
)


def build_extraction_prompt(ocr_text: str, fields: Iterable[FieldSpec]) -> tuple[str, list[str]]:
    fields_list = list(fields)
    if not fields_list:
        raise ValueError("fields must not be empty")

    for f in fields_list:
        if not f.key or not f.display_name:
            raise ValueError(f"invalid field: {f!r}")

    key_order = [f.key for f in fields_list]
    output_template = {k: "" for k in key_order}

    lines: list[str] = []
    lines.append("请从下面OCR文本中抽取指定字段，并按指定JSON格式输出。")
    lines.append("")
    lines.append("【OCR文本】")
    lines.append(ocr_text.strip())
    lines.append("")
    lines.append("【要抽取的字段】")

    for f in fields_list:
        if f.description:
            lines.append(f"- {f.display_name} -> {f.key}（{f.description}）")
        else:
            lines.append(f"- {f.display_name} -> {f.key}")

    lines.append("")
    lines.append("【输出JSON格式】")
    lines.append(json.dumps(output_template, ensure_ascii=False))
    lines.append("")
    lines.append(
        "再次强调：只返回JSON对象本身，不要返回任何额外文字、不要使用Markdown代码块。"
    )

    return "\n".join(lines), key_order


def _post_json(url: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} calling {url}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to call {url}: {e}") from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response from {url}: {body[:500]}") from e


def call_ollama_generate(
    *,
    base_url: str = "http://localhost:11434",
    model: str = "qwen3:4b",
    system: str = _SYSTEM_PROMPT,
    prompt: str,
    temperature: float = 0.0,
    num_predict: int | None = 512,
    enforce_json: bool = True,
    timeout_s: int = 180,
) -> str:
    """Call Ollama /api/generate.

    Returns the model raw text (the `response` field).
    """

    url = base_url.rstrip("/") + "/api/generate"

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    if num_predict is not None:
        payload["options"]["num_predict"] = int(num_predict)

    # Newer Ollama supports format="json" for strict JSON. If not supported, Ollama may error.
    if enforce_json:
        payload["format"] = "json"

    resp = _post_json(url, payload, timeout_s=timeout_s)

    response_text = resp.get("response")
    if response_text is not None and not isinstance(response_text, str):
        raise RuntimeError(f"Unexpected Ollama response schema: {resp}")
    response_text = (response_text or "").strip()

    # Some models (observed with qwen3:4b gguf) may return structured output in
    # `thinking` when `format=json` is set, leaving `response` empty.
    if response_text:
        return response_text

    if enforce_json:
        thinking_text = resp.get("thinking")
        if isinstance(thinking_text, str):
            thinking_text = thinking_text.strip()

            # Fast path: whole thinking is JSON
            if (
                (thinking_text.startswith("{") and thinking_text.endswith("}"))
                or (thinking_text.startswith("[") and thinking_text.endswith("]"))
            ):
                return thinking_text

            # Salvage: extract first {...} or [...] block from thinking (some models
            # may place the actual structured output there while leaving response empty).
            tt = _JSON_FENCE_RE.sub("", thinking_text).strip()
            obj_start, obj_end = tt.find("{"), tt.rfind("}")
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                return tt[obj_start : obj_end + 1].strip()

            arr_start, arr_end = tt.find("["), tt.rfind("]")
            if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                return tt[arr_start : arr_end + 1].strip()

    return ""


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object_only(text: str) -> dict[str, Any]:
    """Parse LLM output that should be a single JSON object.

    Accepts common failure modes like code fences or leading text, then extracts the
    first {...} block.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    s = text.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    # Try to salvage: extract first JSON object boundaries.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")

    candidate = s[start : end + 1].strip()
    return json.loads(candidate)


def parse_json_array_only(text: str) -> list[Any]:
    """Parse LLM output that should be a single JSON array.

    Accepts common failure modes like code fences or leading text, then extracts the
    first [...] block.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    s = text.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # Fast path
    if s.startswith("[") and s.endswith("]"):
        value = json.loads(s)
        if not isinstance(value, list):
            raise ValueError("model output is not a JSON array")
        return value

    # Try to salvage: extract first JSON array boundaries.
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in model output: {text[:300]}")

    candidate = s[start : end + 1].strip()
    value = json.loads(candidate)
    if not isinstance(value, list):
        raise ValueError("model output is not a JSON array")
    return value


def normalize_result(result: dict[str, Any], expected_keys: list[str]) -> dict[str, str]:
    if not isinstance(result, dict):
        raise ValueError("model output is not a JSON object")

    normalized: dict[str, str] = {}
    for k in expected_keys:
        v = result.get(k, "")
        if v is None:
            normalized[k] = ""
        elif isinstance(v, str):
            normalized[k] = v.strip()
        else:
            normalized[k] = str(v)

    return normalized


def extract_fields_from_ocr(
    *,
    ocr_text: str,
    fields: Iterable[FieldSpec],
    base_url: str = "http://localhost:11434",
    model: str = "qwen3:4b",
    temperature: float = 0.0,
    num_predict: int | None = 512,
    enforce_json: bool = True,
) -> dict[str, str]:
    prompt, key_order = build_extraction_prompt(ocr_text, fields)
    raw = call_ollama_generate(
        base_url=base_url,
        model=model,
        prompt=prompt,
        temperature=temperature,
        num_predict=num_predict,
        enforce_json=enforce_json,
    )
    if not raw.strip():
        raise RuntimeError(
            "Ollama returned empty text. If you enabled enforce_json, try enforce_json=false; "
            "also verify base_url/model are correct."
        )
    obj = parse_json_object_only(raw)
    return normalize_result(obj, key_order)
