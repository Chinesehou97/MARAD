import json
import os
from typing import Dict, List, Tuple

from core import config
from storage import memories
from storage import agent_io
from core import llm_pipeline
from core import prompt_builder


AGENT_PREFIXES = ("agent1",)


def _base_agent(agent_tag: str) -> str:
    for prefix in AGENT_PREFIXES:
        if agent_tag.startswith(prefix):
            return prefix
    return ""


def _dedup_lines(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _collect_rules(rule_path: str) -> Dict[str, List[str]]:
    rules_by_agent = {prefix: [] for prefix in AGENT_PREFIXES}
    for r in memories.iter_rules(rule_path):
        if not isinstance(r, dict):
            continue
        agent = (r.get("agent") or "").strip()
        base = _base_agent(agent)
        if not base:
            continue
        line = (r.get("rule_line") or "").strip()
        if not line:
            continue
        rules_by_agent[base].append(line)
    for key in list(rules_by_agent.keys()):
        rules_by_agent[key] = _dedup_lines(rules_by_agent[key])
    return rules_by_agent


def _next_rule_id(prefix: str, used: set) -> str:
    idx = 1
    while True:
        rid = f"{prefix}{idx:03d}"
        if rid not in used:
            used.add(rid)
            return rid
        idx += 1


def _merge_configs(agent_cfgs: List[Tuple[str, Dict]]) -> Dict:
    merged = {
        "derived": {},
        "rules": [],
    }
    seen_ids = set()
    for agent, cfg in agent_cfgs:
        if not isinstance(cfg, dict):
            continue
        rules = cfg.get("rules") or []
        if not isinstance(rules, list):
            continue
        prefix = f"{agent}_"
        for r in rules:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if not rid or rid in seen_ids or not str(rid).startswith(prefix):
                rid = _next_rule_id(prefix, seen_ids)
                r = dict(r)
                r["id"] = rid
            else:
                seen_ids.add(rid)
            merged["rules"].append(r)
    return merged


def translate_rule_memory(
    rule_path: str,
    llm_cfg: Dict,
) -> Tuple[Dict, Dict[str, str]]:
    rules_by_agent = _collect_rules(rule_path)
    agent_cfgs: List[Tuple[str, Dict]] = []
    raw_by_agent: Dict[str, str] = {}

    for agent in AGENT_PREFIXES:
        lines = rules_by_agent.get(agent, [])
        if not lines:
            continue
        prompt = prompt_builder.build_rule_translate_prompt(
            rule_lines=lines,
            agent_tag=agent,
        )
        contents = [{"type": "text", "text": prompt}]
        agent_io.archive_prompt(f"translate_{agent}", prompt)
        raw = llm_pipeline._call_llm(
            contents=contents,
            llm_cfg=llm_cfg,
        )
        raw_by_agent[agent] = raw
        agent_io.archive_raw(f"translate_{agent}", raw)
        try:
            cfg = llm_pipeline._extract_json_config(raw)
        except Exception:
            continue
        agent_cfgs.append((agent, cfg))

    merged = _merge_configs(agent_cfgs)
    return merged, raw_by_agent


def save_rules_config(cfg: Dict, output_dir: str) -> str:
    target_dir = output_dir or config.AGENT_DIR
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "rules.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path
