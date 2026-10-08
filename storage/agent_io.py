import os
from typing import Dict, List

from core import config
from storage import memories


def _archive_io_trace(tag: str, prompt_text: str | None = None, raw_text: str | None = None) -> str:
    """Write a combined input/output trace for a single agent call."""
    io_dir = os.path.join(config.AGENT_DIR, "io")
    os.makedirs(io_dir, exist_ok=True)
    path = os.path.join(io_dir, f"{tag}.txt")

    if prompt_text is None:
        prompt_path = os.path.join(config.AGENT_DIR, "prompt", f"{tag}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()
    if raw_text is None:
        raw_path = os.path.join(config.AGENT_DIR, "raw", f"{tag}.txt")
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== PROMPT =====\n")
        f.write(prompt_text or "")
        f.write("\n\n===== RAW =====\n")
        f.write(raw_text or "")
    return path


def archive_raw(tag: str, raw_text: str) -> str:
    """将 Agent 原始回复写入 AGENT_DIR/raw/{tag}.txt。"""
    raw_dir = os.path.join(config.AGENT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    _archive_io_trace(tag, raw_text=raw_text)
    return path


def archive_prompt(tag: str, prompt_text: str) -> str:
    """将 Agent 输入提示词写入 AGENT_DIR/prompt/{tag}.txt。"""
    prompt_dir = os.path.join(config.AGENT_DIR, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)
    path = os.path.join(prompt_dir, f"{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt_text or "")
    return path


def commit_agent_result(
    tag: str,
    fact_update: Dict,
    rule_lines: List[Dict],
    fact_path: str = None,
    rule_path: str = None,
) -> None:
    """合并写入 FactMemory / RuleMemory。"""
    fact_path = fact_path or config.FACT_MEMORY_PATH
    rule_path = rule_path or config.RULE_MEMORY_PATH

    # FactMemory
    if fact_update:
        old_fact = memories.load_fact(fact_path)
        merged = memories.merge_fact(old_fact, fact_update)
        memories.save_fact(fact_path, merged)

    # RuleMemory
    if rule_lines:
        memories.append_rules(rule_path, rule_lines)

    # 预留：tag 目前仅用于 raw 存档命名
