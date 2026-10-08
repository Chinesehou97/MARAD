import os
import re
import sys
from typing import Dict, List, Tuple

from core import config
from agents import context_builder
from core import llm_pipeline
from storage import agent_io
from storage import memories
from agents import agent1_census


def _log_agent_done(tag: str) -> None:
    print(f"[AGENT DONE] {tag}", file=sys.stderr, flush=True)


def _pluralize_base(base: str) -> str:
    if not base:
        return base
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return f"{base}es"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return f"{base[:-1]}ies"
    return f"{base}s"


def _normalize_rule_lines(rule_lines: List[Dict], base_classes: List[str]) -> List[Dict]:
    if not rule_lines or not base_classes:
        return rule_lines
    plural_map = {}
    for base in base_classes:
        if not base or not isinstance(base, str):
            continue
        plural = _pluralize_base(base)
        if plural and plural != base:
            plural_map[plural] = base
    if not plural_map:
        return rule_lines

    normalized = []
    for rule in rule_lines:
        if not isinstance(rule, dict):
            normalized.append(rule)
            continue
        line = rule.get("rule_line")
        if not isinstance(line, str) or not line:
            normalized.append(rule)
            continue
        new_line = line
        for plural, base in plural_map.items():
            new_line = re.sub(rf"\b{re.escape(plural)}\b", base, new_line)
        if new_line != line:
            updated = dict(rule)
            updated["rule_line"] = new_line
            normalized.append(updated)
        else:
            normalized.append(rule)
    return normalized


def _commit_agent_result(tag: str, fact_update: Dict, rule_lines: List[Dict], base_classes: List[str]) -> None:
    agent_io.commit_agent_result(tag, fact_update, _normalize_rule_lines(rule_lines, base_classes))


def _strip_sometimes_rules(rule_lines: List[Dict]) -> List[Dict]:
    if not rule_lines:
        return rule_lines
    cleaned = []
    for rule in rule_lines:
        if not isinstance(rule, dict):
            cleaned.append(rule)
            continue
        line = rule.get("rule_line")
        if isinstance(line, str) and line.strip().lower().startswith("each sample has "):
            continue
        cleaned.append(rule)
    return cleaned


def _normalize_class_list(raw) -> List[str]:
    names: List[str] = []
    seen = set()

    def _add(name: str):
        cleaned = name.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        names.append(cleaned)

    def _walk(item):
        if item is None:
            return
        if isinstance(item, str):
            _add(item)
            return
        if isinstance(item, dict):
            for key in ("base_class", "class", "name", "label", "type"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    _add(val)
                    return
            for val in item.values():
                if isinstance(val, str) and val.strip():
                    _add(val)
                    return
            for val in item.values():
                _walk(val)
            return
        if isinstance(item, (list, tuple, set)):
            for val in item:
                _walk(val)
            return

    _walk(raw)
    return names


def _coerce_agent1_fact(fact_update: Dict) -> Dict:
    if not isinstance(fact_update, dict):
        return fact_update
    for key in ("stable", "always_present", "sometimes_present"):
        raw = fact_update.get(key, [])
        normalized = _normalize_class_list(raw)
        if raw != normalized:
            fact_update[f"{key}_raw"] = raw
            fact_update[key] = normalized
    return fact_update


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (TypeError, ValueError):
            return default
    return default


def _build_sometimes_or_rules(fact_memory: Dict, agent_tag: str) -> List[Dict]:
    counts_by_image = fact_memory.get("counts_sometimes_by_image", {})
    sometimes = fact_memory.get("sometimes_present", []) or []
    if not isinstance(counts_by_image, dict) or not sometimes:
        return []

    image_ids = sorted(counts_by_image.keys(), key=lambda x: str(x))
    if not image_ids:
        return []

    options: List[Tuple[str, int]] = []
    for base in sometimes:
        if not isinstance(base, str) or not base.strip():
            continue
        counts = []
        present_counts = []
        for img_id in image_ids:
            per_image = counts_by_image.get(img_id) or {}
            if not isinstance(per_image, dict):
                per_image = {}
            count_val = _safe_int(per_image.get(base, 0), default=0)
            counts.append(count_val)
            if count_val > 0:
                present_counts.append(count_val)
        if not present_counts:
            continue
        if all(c > 0 for c in counts):
            continue
        if len(set(present_counts)) != 1:
            continue
        options.append((base, present_counts[0]))

    if not options:
        return []

    parts = [f"{count} {base}" for base, count in options]
    if len(parts) == 1:
        rule_line = f"each sample has {parts[0]}"
    else:
        rule_line = "each sample has " + " or ".join(parts)
    return [{"rule_line": rule_line, "agent": agent_tag}]


def run_agent1(store_data: Dict, llm_cfg: Dict) -> Dict:
    base_classes = store_data.get("base_classes") or sorted(store_data.get("by_base_image", {}).keys())
    if not base_classes:
        print("Agent-1: no base_classes, skip.")
        return memories.load_fact(config.FACT_MEMORY_PATH)

    for idx, base_class in enumerate(base_classes):
        ctx = context_builder.build_context("agent1", store_data, {}, batch_spec=[base_class])
        tag = f"agent1_b{idx}"
        prompt = agent1_census.build_prompt(ctx, agent_tag=tag)
        agent_io.archive_prompt(tag, prompt)
        fact_update, rule_lines, raw = llm_pipeline.request_agent_output(
            prompt_text=prompt,
            tag=tag,
            llm_cfg=llm_cfg,
        )
        fact_update = _coerce_agent1_fact(fact_update)
        agent_io.archive_raw(tag, raw)
        rule_lines = _strip_sometimes_rules(rule_lines)
        _commit_agent_result(tag, fact_update, rule_lines, base_classes)
        _log_agent_done(tag)

    fact_memory = memories.load_fact(config.FACT_MEMORY_PATH)
    if fact_memory.get("sometimes_present") or context_builder._has_subtypes(store_data):
        tag = "agent1_counts"
        ctx = context_builder.build_context("agent1_counts", store_data, fact_memory, batch_spec=None)
        prompt = agent1_census.build_prompt_counts(ctx, agent_tag=tag)
        agent_io.archive_prompt(tag, prompt)
        fact_update, rule_lines, raw = llm_pipeline.request_agent_output(
            prompt_text=prompt,
            tag=tag,
            llm_cfg=llm_cfg,
        )
        agent_io.archive_raw(tag, raw)
        _commit_agent_result(tag, fact_update, [], base_classes)
        _log_agent_done(tag)
        fact_memory = memories.load_fact(config.FACT_MEMORY_PATH)
        or_rules = _build_sometimes_or_rules(fact_memory, tag)
        if or_rules:
            _commit_agent_result(tag, {}, or_rules, base_classes)
    else:
        print("Agent-1 counts: no sometimes_present or subtypes, skip.")
    return fact_memory


def print_rules_by_agent(rule_path: str):
    from storage.memories import iter_rules

    if not os.path.exists(rule_path):
        print("No rule memory found.")
        return
    print("\n===== RULES (by agent) =====")
    for r in iter_rules(rule_path):
        try:
            agent = r.get("agent")
            if agent:
                print(f"- agent={agent}, rule_line={r.get('rule_line')}, evidence={r.get('evidence')}")
            else:
                print(f"- rule_line={r.get('rule_line')}, evidence={r.get('evidence')}")
        except Exception:
            continue
