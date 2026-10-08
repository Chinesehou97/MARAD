import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _call_remote_api(
    contents: List[Dict],
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": contents}],
        temperature=temperature,
        extra_body={"enable_thinking": True},
        stream=True,
    )
    chunks = []
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            chunks.append(delta.content)
    return "".join(chunks)


def _call_llm(contents: List[Dict], llm_cfg: Optional[Dict] = None, *,
              model: Optional[str] = None, base_url: str = "", api_key: str = "",
              temperature: float = 0) -> str:
    """Call the remote API; credentials and provider settings come from the environment."""
    from core import config
    cfg = dict(llm_cfg or {})
    if model is not None:
        cfg["model"] = model
    cfg.setdefault("model", config.LLM_MODEL_DEFAULT)
    cfg.setdefault("base_url", base_url or config.LLM_BASE_URL_DEFAULT)
    cfg.setdefault("api_key", api_key or os.getenv(config.LLM_API_KEY_ENV, ""))
    cfg.setdefault("temperature", temperature)
    required = {"model": "MARAD_LLM_MODEL", "base_url": "MARAD_LLM_BASE_URL",
                "api_key": config.LLM_API_KEY_ENV}
    missing = [env_name for field, env_name in required.items() if not cfg.get(field)]
    if missing:
        raise ValueError("Missing remote API configuration: " + ", ".join(missing))
    return _call_remote_api(contents, str(cfg["model"]), str(cfg["base_url"]),
                            str(cfg["api_key"]), float(cfg.get("temperature", 0) or 0))


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)


def _slice_balanced_json(text: str, open_ch: str, close_ch: str) -> str:
    start = text.find(open_ch)
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _iter_json_object_candidates(text: str) -> List[str]:
    cleaned = _strip_think_blocks(text).strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I):
        block = match.group(1).strip()
        if block:
            candidates.append(block)
    balanced = _slice_balanced_json(cleaned, "{", "}")
    if balanced:
        candidates.append(balanced)

    seen = set()
    out: List[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_json_config(text: str) -> Dict:
    """
    从 LLM 回复中提取 JSON 对象：
    - 支持纯 JSON
    - 支持 ```json 代码块
    - 支持带有 <think>...</think> 的 Thinking 输出
    """
    last_error: Optional[Exception] = None
    for candidate in _iter_json_object_candidates(text):
        try:
            data = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = ValueError("解析结果不是 JSON 对象")
    raise ValueError(f"无法从 LLM 输出中提取 JSON 对象: {last_error}")


def _extract_agent_payload(raw_text: str) -> Tuple[Dict, List[Dict]]:
    """
    解析 Agent 输出：
    - 要求包含 FACT_UPDATE: <json object> 和 RULE_LINES: <json array>
    - 兜底1：取最后一对 FACT_UPDATE/RULE_LINES 解析
    - 兜底2：括号配对截取 JSON 块解析
    - 若未出现关键字，则改用 json 代码块解析
    """

    def _find_last_pair(text: str) -> Tuple[int, int]:
        fact_positions = [m.start() for m in re.finditer(r"FACT_UPDATE:", text)]
        rule_positions = [m.start() for m in re.finditer(r"RULE_LINES:", text)]
        if not fact_positions or not rule_positions:
            return -1, -1
        for fact_pos in reversed(fact_positions):
            later_rules = [p for p in rule_positions if p > fact_pos]
            if later_rules:
                return fact_pos, later_rules[-1]
        return -1, -1

    def _is_placeholder(text: str) -> bool:
        if not text:
            return False
        return "..." in text

    def _slice_json(text: str, start: int, open_ch: str, close_ch: str) -> str:
        idx = text.find(open_ch, start)
        if idx == -1:
            raise ValueError(f"missing {open_ch}")
        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[idx : i + 1]
        raise ValueError(f"unbalanced {open_ch}{close_ch}")

    def _extract_json_codeblocks(text: str) -> List[str]:
        blocks = []
        for m in re.finditer(r"```json\s*(.*?)```", text, flags=re.S | re.I):
            blocks.append(m.group(1).strip())
        return blocks

    def _parse_rule_lines(raw: str) -> List[Dict]:
        if _is_placeholder(raw):
            raise ValueError("placeholder output detected")
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise ValueError("RULE_LINES is not a JSON array")
        return data

    def _salvage_rule_lines(text: str, rule_pos: int) -> List[Dict]:
        if rule_pos != -1:
            try:
                rule_str = text[rule_pos + len("RULE_LINES:") :].strip()
                return _parse_rule_lines(rule_str)
            except Exception:
                pass
            try:
                rule_str = _slice_json(text, rule_pos + len("RULE_LINES:"), "[", "]")
                return _parse_rule_lines(rule_str)
            except Exception:
                pass
        try:
            blocks = _extract_json_codeblocks(text)
            for block in reversed(blocks):
                try:
                    return _parse_rule_lines(block)
                except Exception:
                    continue
        except Exception:
            pass
        return []

    fact_update: Dict = {}
    rule_lines: List[Dict] = []
    text = raw_text or ""
    err_fact = "missing FACT_UPDATE block"
    err_rule = "missing RULE_LINES block"

    def _append_failure(detail: str) -> None:
        nonlocal fact_update
        if not isinstance(fact_update, dict):
            fact_update = {}
        failures = fact_update.get("agent_failures")
        if not isinstance(failures, list):
            failures = []
        failures.append(
            {
                "reason": "AGENT_OUTPUT_PARSE_FAILED",
                "detail": detail[:200],
            }
        )
        fact_update["agent_failures"] = failures

    def _parse_fact() -> Dict:
        nonlocal err_fact
        if fact_pos != -1:
            if rule_pos != -1 and rule_pos > fact_pos:
                fact_str = text[fact_pos + len("FACT_UPDATE:") : rule_pos].strip()
                if fact_str:
                    try:
                        if _is_placeholder(fact_str):
                            raise ValueError("placeholder output detected")
                        data = json.loads(fact_str)
                        if not isinstance(data, dict):
                            raise ValueError("FACT_UPDATE is not a JSON object")
                        err_fact = ""
                        return data
                    except Exception as e:
                        err_fact = str(e)
            try:
                fact_str = _slice_json(text, fact_pos + len("FACT_UPDATE:"), "{", "}")
                if _is_placeholder(fact_str):
                    raise ValueError("placeholder output detected")
                data = json.loads(fact_str) if fact_str else {}
                if not isinstance(data, dict):
                    raise ValueError("FACT_UPDATE is not a JSON object")
                err_fact = ""
                return data
            except Exception as e:
                err_fact = str(e)
        try:
            blocks = _extract_json_codeblocks(text)
            if blocks:
                fact_str = blocks[0]
                if _is_placeholder(fact_str):
                    raise ValueError("placeholder output detected")
                data = json.loads(fact_str) if fact_str else {}
                if not isinstance(data, dict):
                    raise ValueError("FACT_UPDATE is not a JSON object")
                err_fact = ""
                return data
        except Exception as e:
            err_fact = str(e)
        return {}

    def _parse_rules() -> List[Dict]:
        nonlocal err_rule
        if rule_pos != -1:
            try:
                rule_str = text[rule_pos + len("RULE_LINES:") :].strip()
                data = _parse_rule_lines(rule_str)
                err_rule = ""
                return data
            except Exception as e:
                err_rule = str(e)
            try:
                rule_str = _slice_json(text, rule_pos + len("RULE_LINES:"), "[", "]")
                data = _parse_rule_lines(rule_str)
                err_rule = ""
                return data
            except Exception as e:
                err_rule = str(e)
            data = _salvage_rule_lines(text, rule_pos)
            if data:
                err_rule = ""
            return data
        try:
            blocks = _extract_json_codeblocks(text)
            if len(blocks) >= 2:
                data = _parse_rule_lines(blocks[1])
                err_rule = ""
                return data
        except Exception as e:
            err_rule = str(e)
        return []

    fact_pos, rule_pos = _find_last_pair(text)
    fact_update = _parse_fact()
    rule_lines = _parse_rules()

    if err_fact or err_rule:
        detail_parts = []
        if err_fact:
            detail_parts.append(f"FACT_UPDATE: {err_fact}")
        if err_rule:
            detail_parts.append(f"RULE_LINES: {err_rule}")
        detail = "; ".join(detail_parts)
        if fact_update:
            _append_failure(detail)
        else:
            fact_update = {
                "agent_failures": [
                    {
                        "reason": "AGENT_OUTPUT_PARSE_FAILED",
                        "detail": detail[:200],
                    }
                ]
            }
    return fact_update, rule_lines


def _normalize_agent_output(
    fact_update: Dict,
    rule_lines: List[Dict],
    agent_tag: str,
) -> Tuple[Dict, List[Dict]]:
    if not isinstance(fact_update, dict):
        fact_update = {}
    if agent_tag:
        fact_update["agent"] = agent_tag
    cleaned_rules: List[Dict] = []
    for r in rule_lines:
        if not isinstance(r, dict):
            continue
        item = dict(r)
        item.pop("step", None)
        if agent_tag:
            item["agent"] = agent_tag
        cleaned_rules.append(item)
    return fact_update, cleaned_rules


def request_agent_output(prompt_text: str, tag: str, llm_cfg: Optional[Dict] = None) -> Tuple[Dict, List[Dict], str]:
    """Send text evidence to the remote API and parse FactMemory and RuleMemory updates."""
    raw_text = _call_llm(contents=[{"type": "text", "text": prompt_text}], llm_cfg=llm_cfg)
    fact_update, rule_lines = _extract_agent_payload(raw_text)
    fact_update, rule_lines = _normalize_agent_output(fact_update, rule_lines, tag)
    return fact_update, rule_lines, raw_text
