import os
import json
from typing import List, Dict


def finalize_constraints(rule_memory_path: str, include_fail: bool = False) -> str:
    """
    读取 rule_memory.jsonl，按 agent 顺序输出规则行。
    include_fail 控制是否输出 FAIL 的规则行。
    """
    if not os.path.exists(rule_memory_path):
        return "No rules found."

    rules: List[Dict] = []
    with open(rule_memory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rules.append(r)
            except Exception:
                continue

    def _agent_sort_key(agent: str):
        if not agent:
            return (99, 99, 99, "")
        if agent.startswith("agent1"):
            return (0, 0, 0, agent)
        return (98, 0, 0, agent)

    rules = [
        (idx, r)
        for idx, r in enumerate(rules)
    ]
    rules.sort(key=lambda item: (_agent_sort_key(item[1].get("agent")), item[0]))

    out_lines = []
    for _, r in rules:
        if not str(r.get("agent") or "").startswith("agent1"):
            continue
        rule_line = r.get("rule_line")
        rule_text = str(rule_line or "")
        if (not include_fail) and rule_text.strip().upper().startswith("FAIL"):
            continue
        agent = r.get("agent")
        if agent:
            out_lines.append(f"agent={agent} | {rule_line}")
        else:
            out_lines.append(f"{rule_line}")
    return "\n".join(out_lines)
