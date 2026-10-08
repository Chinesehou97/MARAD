import json
import os
import shutil
from typing import Dict, List, Iterator, Callable, Optional

from core import config


# -------------------- FactMemory --------------------

def load_fact(path: str) -> Dict:
    """加载结构化中间结论；文件不存在时返回空字典。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_memories() -> None:
    """清空 FactMemory / RuleMemory 以及 raw/prompt 目录，确保每次运行不读取旧文件。"""
    for p in [config.FACT_MEMORY_PATH, config.RULE_MEMORY_PATH]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    raw_dir = os.path.join(config.AGENT_DIR, "raw")
    prompt_dir = os.path.join(config.AGENT_DIR, "prompt")
    if os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir, ignore_errors=True)
    if os.path.isdir(prompt_dir):
        shutil.rmtree(prompt_dir, ignore_errors=True)


def _merge_lists(old_list: List, new_list: List) -> List:
    seen = set()
    merged = []
    for item in old_list + new_list:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def merge_fact(old: Dict, update: Dict) -> Dict:
    """
    递归合并中间结论：
    - dict 递归合并
    - list 去重，旧在前，新补后
    - 其他冲突保留旧值
    """
    if not isinstance(old, dict) or not isinstance(update, dict):
        return old

    merged = dict(old)
    for k, v in update.items():
        if k in merged:
            if k == "agent":
                merged[k] = v
                continue
            old_v = merged[k]
            if isinstance(old_v, dict) and isinstance(v, dict):
                merged[k] = merge_fact(old_v, v)
            elif isinstance(old_v, list) and isinstance(v, list):
                merged[k] = _merge_lists(old_v, v)
            else:
                merged[k] = old_v
        else:
            merged[k] = v
    return merged


def save_fact(path: str, fact: Dict) -> None:
    """保存结构化中间结论。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fact, f, indent=2, ensure_ascii=False)


# -------------------- RuleMemory --------------------

def append_rules(path: str, rules: List[Dict], validator: Optional[Callable[[Dict], bool]] = None) -> None:
    """
    追加写入规则行（去重）：
    - 去重 key = (agent, rule_line)；无 agent 则用 (rule_line)
    - 可选 validator(rule_dict)->bool 过滤非法 rule_line
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    agent = item.get("agent")
                    if agent:
                        key = (agent, item.get("rule_line"))
                    else:
                        key = (item.get("rule_line"),)
                    seen.add(key)
                except Exception:
                    continue

    with open(path, "a", encoding="utf-8") as f:
        for r in rules:
            agent = r.get("agent")
            if agent:
                key = (agent, r.get("rule_line"))
            else:
                key = (r.get("rule_line"),)
            if key in seen:
                continue
            if validator is not None and not validator(r):
                continue
            seen.add(key)
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def iter_rules(path: str) -> Iterator[Dict]:
    """迭代已有规则行；文件不存在则返回空迭代器。"""
    if not os.path.exists(path):
        return iter(())

    def _gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    return _gen()
