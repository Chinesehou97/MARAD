import math
import re
from typing import Any, Dict, List, Optional, Set


def _strip_suffix_digits(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def _parse_label_parts(text: str):
    """
    解析标签/实例号：
      - "red wire1" / "wire_red3" / "wire3" / "wire"
    返回 (base_class, subtype, instance_id)
    """
    if not text:
        return "", None, None

    inst_id = None
    base = text
    subtype = None

    if " " in text:
        parts = text.split()
        base_token = parts[-1]
        subtype = " ".join(parts[:-1]).strip() or None
        m = re.match(r"(.+?)(\d+)$", base_token)
        if m:
            base = m.group(1)
            inst_id = m.group(2)
        else:
            base = base_token
    elif "_" in text:
        parts = text.split("_")
        base_token = parts[-1]
        subtype = "_".join(parts[:-1]).strip() or None
        m = re.match(r"(.+?)(\d+)$", base_token)
        if m:
            base = m.group(1)
            inst_id = m.group(2)
        else:
            base = base_token
    else:
        m = re.match(r"(.+?)(\d+)$", text)
        if m:
            base = m.group(1)
            inst_id = m.group(2)
        else:
            base = text

    base = _strip_suffix_digits(base)
    return base, subtype, inst_id


def _parse_detection(det: Dict) -> Dict:
    """
    将原始 detection 统一转成内部使用的结构：
    - raw_label: 原始标签（优先 tag，其次 label）
    - base_class: 去掉数字后得到的基础类别名
    - instance_id: 尾部数字（如果有）
    - bbox / cx / cy / diag: 外接框与中心、对角线长度
    """
    raw_label = det.get("tag") or det.get("label") or ""
    base_class, subtype, inst_id = _parse_label_parts(raw_label or det.get("label") or "")

    bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    diag = det.get("diag")
    if diag is None:
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        diag = math.hypot(w, h)

    return {
        "raw_label": raw_label,
        "base_class": base_class,
        "instance_id": inst_id,
        "subtype": det.get("subtype") or subtype,
        "bbox": bbox,
        "cx": cx,
        "cy": cy,
        "diag": diag,
    }


def _point_in_box(cx: float, cy: float, box: List[float]) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_RE = re.compile(
    rf"\b(?P<name>diag)\s+in\s*"
    rf"(?P<left>[\[\(])\s*(?P<low>{_NUMBER_RE})\s*,\s*"
    rf"(?P<high>{_NUMBER_RE})\s*(?P<right>[\]\)])"
)


def _normalize_filter_expr(expr: str) -> str:
    """Accept both JSON-rule boolean operators and Python boolean operators."""
    normalized = str(expr)

    def _replace_interval(match: re.Match) -> str:
        name = match.group("name")
        low_op = ">=" if match.group("left") == "[" else ">"
        high_op = "<=" if match.group("right") == "]" else "<"
        return (
            f"({name} {low_op} {match.group('low')} "
            f"and {name} {high_op} {match.group('high')})"
        )

    normalized = _INTERVAL_RE.sub(_replace_interval, normalized)
    normalized = normalized.replace("&&", " and ")
    normalized = normalized.replace("||", " or ")
    normalized = re.sub(r"!(?!=)", " not ", normalized)
    return normalized.strip()


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, str):
        try:
            expected = float(expected)
        except ValueError:
            pass
    return actual == expected


def _eval_filter_dict(expr: Dict[str, Any], env: Dict[str, object]) -> bool:
    allowed_names = {"diag", "subtype", "base_class"}
    for key, expected in expr.items():
        if key not in allowed_names:
            raise ValueError("bad filter object key")
        actual = env.get(key)
        if isinstance(expected, list):
            if not any(_values_equal(actual, item) for item in expected):
                return False
        else:
            if not _values_equal(actual, expected):
                return False
    return True


def _filter_env(det: Dict) -> Dict[str, object]:
    return {
        "diag": det["diag"],
        "subtype": det["subtype"],
        "base_class": det["base_class"],
    }


def _det_matches_rule_target(det: Dict, target_cls: str, filt: Any = None) -> bool:
    if isinstance(filt, dict) and filt.get("base_class"):
        return True
    return det["base_class"] == target_cls or det["raw_label"] == target_cls


def _eval_filter(expr: Optional[Any], env: Dict[str, object]) -> bool:
    """
    对规则里的 filter 表达式做安全求值：
    - 允许的变量名：diag / subtype / base_class
    - 允许：and / or / not，以及 ==, !=, <, <=, >, >=
    - 不允许函数调用、属性访问、算术运算等
    """
    if not expr:
        return True
    if isinstance(expr, dict):
        return _eval_filter_dict(expr, env)
    expr = _normalize_filter_expr(expr)

    import ast

    allowed_names = {"diag", "subtype", "base_class"}
    ignorable_stat_names = {"min", "max", "mean", "std", "median", "low", "high"}

    class _UnsupportedFilterClause(Exception):
        pass

    def _has_only_ignorable_unknown_names(node) -> bool:
        unknown_names = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id not in allowed_names
        }
        return bool(unknown_names) and unknown_names <= ignorable_stat_names

    def _eval(node):
        if isinstance(node, ast.BoolOp):
            vals = []
            for value in node.values:
                try:
                    vals.append(_eval(value))
                except _UnsupportedFilterClause:
                    continue
            if not vals:
                return True
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            raise ValueError("bad boolop")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _eval(node.operand)

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval(comp)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                else:
                    raise ValueError("bad compare op")
                if not ok:
                    return False
                left = right
            return True

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                if node.id in ignorable_stat_names:
                    raise _UnsupportedFilterClause("unsupported stat name")
                raise ValueError("bad name")
            return env.get(node.id)

        if isinstance(node, ast.Constant):
            return node.value

        if _has_only_ignorable_unknown_names(node):
            raise _UnsupportedFilterClause("unsupported stat expr")
        raise ValueError("bad expr")

    tree = ast.parse(expr, mode="eval")
    try:
        return bool(_eval(tree.body))
    except _UnsupportedFilterClause:
        return True


def _validate_config_new(cfg: Dict) -> List[str]:
    """
    对 LLM 生成的 config（derived + rules）做结构校验：
    - 禁止 None 和形如 <...> 的占位符
    - 检查字段名和 rule.type 合法性
    - 提前解析 filter，防止运行时崩溃
    """
    errs: List[str] = []

    if not isinstance(cfg, dict):
        return ["config must be object"]

    def _has_none(obj) -> bool:
        if obj is None:
            return True
        if isinstance(obj, dict):
            return any(_has_none(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_none(v) for v in obj)
        return False

    def _has_placeholder(obj) -> bool:
        if isinstance(obj, str):
            return bool(re.search(r"<[^>]+>", obj))
        if isinstance(obj, dict):
            return any(_has_placeholder(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_placeholder(v) for v in obj)
        return False

    if _has_none(cfg):
        errs.append("config contains null value")
    if _has_placeholder(cfg):
        errs.append("config contains placeholder tokens")

    allowed_top = {"derived", "rules"}
    for k in cfg.keys():
        if k not in allowed_top:
            errs.append(f"unknown top-level field {k}")

    rules = cfg.get("rules", [])
    if not isinstance(rules, list):
        errs.append("rules must be list")
        return errs

    def _check_allowed_fields(obj: Dict, allowed: Set[str], ctx: str):
        for k in obj.keys():
            if k not in allowed:
                errs.append(f"{ctx}: unknown field {k}")

    for r in rules:
        if not isinstance(r, dict):
            errs.append("rule must be object")
            continue
        rid = r.get("id", "?")
        rtype = r.get("type")
        if rtype not in {"count", "imply"}:
            errs.append(f"rule {rid}: invalid type {rtype}")
            continue

        if rtype == "count":
            _check_allowed_fields(
                r,
                {"id", "type", "container", "target", "subtype", "filter", "exact", "min", "max"},
                f"rule {rid}",
            )
            if not r.get("target"):
                errs.append(f"rule {rid}: count missing target")
            cont = r.get("container", "whole_image")
            if not (cont == "whole_image" or (isinstance(cont, dict) and "inside" in cont)):
                errs.append(f"rule {rid}: invalid container")
            if r.get("filter"):
                try:
                    _eval_filter(
                        r.get("filter"),
                        {"diag": 0, "subtype": "", "base_class": ""},
                    )
                except Exception:
                    errs.append(f"rule {rid}: invalid filter")

        elif rtype == "imply":
            _check_allowed_fields(
                r,
                {"id", "type", "if", "then"},
                f"rule {rid}",
            )
            if not isinstance(r.get("if"), list) or not isinstance(r.get("then"), list):
                errs.append(f"rule {rid}: imply requires if/then lists")
            else:
                for part in ("if", "then"):
                    errs.extend(f"rule {rid} {part}: {err}" for err in _validate_config_new({"rules": r[part]}))
    derived = cfg.get("derived", {})
    if not isinstance(derived, dict):
        errs.append("derived must be object")
    elif any(derived.values()):
        errs.append("derived features are not supported by the Counting Agent")
    return errs


def evaluate_rules_structured(detections: List[Dict], config: Dict) -> Dict:
    """
    新规则执行入口：
    - detections：检测 JSON 中的对象列表
    - config：LLM 生成的 {derived, rules}
    支持 count，以及 Counting Agent 数量选项使用的 imply。
    """
    errs = _validate_config_new(config)
    if errs:
        return {
            "overall_pass": False,
            "failed_rules": [
                {
                    "rule_id": "config",
                    "type": "schema",
                    "message": "; ".join(errs),
                    "evidence": {},
                }
            ],
            "passed_rules_count": 0,
            "failed_rules_count": 1,
        }

    parsed = [_parse_detection(det) for det in detections]

    failed = []
    passed_count = 0

    def _match_container(item: Dict, container) -> bool:
        """
        container:
        - "whole_image": 不限制
        - {"inside": "<base_class>"}: 要求 item 中心点落在该 base_class 的某个实例 bbox 内
        """
        if container == "whole_image":
            return True
        if isinstance(container, dict) and "inside" in container:
            ref_cls = container["inside"]
            for ref in parsed:
                if ref["base_class"] != ref_cls:
                    continue
                if _point_in_box(item["cx"], item["cy"], ref["bbox"]):
                    return True
            return False
        return False

    def add_fail(rid, rtype, msg, evidence):
        failed.append({"rule_id": rid, "type": rtype, "message": msg, "evidence": evidence})

    for r in config.get("rules", []):
        rtype = r.get("type")
        rid = r.get("id", "?")

        if rtype == "count":
            target_cls = r.get("target")
            subtype = r.get("subtype")
            container = r.get("container", "whole_image")
            filt = r.get("filter")
            exact = r.get("exact")
            min_v = r.get("min")
            max_v = r.get("max")
            if not target_cls:
                add_fail(rid, rtype, "COUNT missing target", {})
                continue

            candidates = []
            for det in parsed:
                if not _det_matches_rule_target(det, target_cls, filt):
                    continue
                if subtype and det["subtype"] != subtype:
                    continue
                if not _match_container(det, container):
                    continue
                env = _filter_env(det)
                try:
                    if not _eval_filter(filt, env):
                        continue
                except Exception:
                    add_fail(rid, rtype, "COUNT failed: invalid filter", {})
                    candidates = None
                    break
                candidates.append(det)
            if candidates is None:
                continue

            cnt = len(candidates)
            ok = True
            if exact is not None:
                ok = cnt == exact
            else:
                if min_v is not None and cnt < min_v:
                    ok = False
                if max_v is not None and cnt > max_v:
                    ok = False

            if ok:
                passed_count += 1
            else:
                msg = f"COUNT failed: target={target_cls}"
                if exact is not None:
                    msg += f" expected exact={exact}"
                else:
                    msg += f" expected min={min_v} max={max_v}"
                msg += f" found={cnt}"
                add_fail(
                    rid,
                    rtype,
                    msg,
                    {"found_instances": [d["raw_label"] for d in candidates]},
                )

        elif rtype == "imply":
            sub_if = evaluate_rules_structured(
                detections,
                {"derived": config.get("derived", {}), "rules": r.get("if", [])},
            )
            if sub_if["overall_pass"]:
                sub_then = evaluate_rules_structured(
                    detections,
                    {"derived": config.get("derived", {}), "rules": r.get("then", [])},
                )
                if sub_then["overall_pass"]:
                    passed_count += 1
                else:
                    add_fail(
                        rid,
                        rtype,
                        "IMPLY triggered but THEN failed",
                        {"failed_then": sub_then["failed_rules"]},
                    )
            else:
                # IF 不满足，则整条 imply 视为“真”（不产生约束）
                passed_count += 1

    overall_pass = len(failed) == 0
    return {
        "overall_pass": overall_pass,
        "failed_rules": failed,
        "passed_rules_count": passed_count,
        "failed_rules_count": len(failed),
    }
