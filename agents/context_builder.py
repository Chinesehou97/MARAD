from typing import Dict, List

from storage import store


def _has_subtypes(store_data: Dict) -> bool:
    for rec in (store_data.get("inst_lookup") or {}).values():
        subtype = rec.get("subtype")
        if subtype and str(subtype).strip():
            return True
    return False


def _format_agent1_labels(store_data: Dict, image_ids: List[str], base_classes: List[str] | None = None) -> str:
    base_set = set(base_classes or [])
    lines: List[str] = []
    for img in image_ids:
        labels = store.get_labels(store_data, img)
        lines.append(f"- image_id: {img}")
        lines.append("  labels:")
        for lbl in labels:
            if base_set and lbl.get("base_class") not in base_set:
                continue
            lines.append(f"    - base_class: {lbl.get('base_class','')}, num: {lbl.get('num','')}")
    return "\n".join(lines)


def _format_agent1_counts(store_data: Dict, fact_memory: Dict, image_ids: List[str]) -> str:
    lines: List[str] = []
    sometimes = fact_memory.get("sometimes_present", [])
    lines.append("# sometimes_present")
    lines.append(", ".join(sometimes) if sometimes else "[]")
    lines.append("")
    lines.append("# labels_by_image")
    for img in image_ids:
        labels = store.get_labels(store_data, img)
        lines.append(f"- image_id: {img}")
        lines.append("  labels:")
        for lbl in labels:
            base_class = lbl.get("base_class", "")
            subtype = lbl.get("subtype")
            num = lbl.get("num", "")
            if subtype:
                lines.append(f"    - base_class: {base_class}, subtype: {subtype}, num: {num}")
            else:
                lines.append(f"    - base_class: {base_class}, num: {num}")
    return "\n".join(lines)


def build_context(agent_type: str, store_data: Dict, fact_memory: Dict, batch_spec) -> str:
    """Build Counting Agent evidence from the same normal reference images."""
    image_ids = store_data.get("image_ids") or sorted(store_data.get("by_image_base", {}).keys())
    if agent_type == "agent1":
        return _format_agent1_labels(store_data, image_ids, list(batch_spec) if batch_spec else None)
    if agent_type == "agent1_counts":
        return _format_agent1_counts(store_data, fact_memory, image_ids)
    raise ValueError(f"Unsupported agent type: {agent_type}")
