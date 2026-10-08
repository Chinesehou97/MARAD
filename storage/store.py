import json
import os
import re
from typing import Dict, List


def _normalize_base_class(name: str) -> str:
    """Drop trailing digits to keep the core class name (e.g., liquid1 -> liquid)."""
    if not name:
        return ""
    return re.sub(r"\d+$", "", name)


def _normalize_full_name(label: str) -> str:
    """
    Normalize the full name by stripping trailing digits from the last token.
    Examples: "banana18" -> "banana", "red wire1" -> "red wire".
    """
    parts = (label or "").split()
    if not parts:
        return ""
    last = _normalize_base_class(parts[-1])
    if not last:
        last = parts[-1]
    parts[-1] = last
    return " ".join(p for p in parts if p).strip()


def _split_label(label: str) -> Dict[str, str]:
    """
    Split a full label into base_class / subtype.
    - base_class: last word with trailing digits removed
    - subtype: everything before the last word (can be empty)
    """
    parts = (label or "").split()
    if not parts:
        return {"base_class": "", "subtype": ""}
    raw_base = parts[-1]
    base_class = _normalize_base_class(raw_base)
    subtype = " ".join(parts[:-1])
    return {"base_class": base_class, "subtype": subtype}


def build_store(
    results: List[Dict],
) -> Dict:
    """
    Build an indexable store from detection JSON (no truncation).
    - image_id: re-numbered per run as 1..N in input order (ignore original path/id)
    - num: counts from 1 within each image_id + base_class
    - inst_lookup key: image_id#base_class#num to avoid collisions
    """
    inst_lookup: Dict[str, Dict] = {}
    by_image_base: Dict[str, Dict[str, List[str]]] = {}
    by_base_image: Dict[str, Dict[str, List[str]]] = {}

    image_ids: List[str] = []
    image_id_map: Dict[str, str] = {}
    base_classes = set()
    img_class_counters: Dict[str, Dict[str, int]] = {}

    for res in results:
        orig_image_id = res.get("image_id") or os.path.splitext(os.path.basename(res.get("image_path", "")))[0]
        if orig_image_id not in image_id_map:
            image_id_map[orig_image_id] = str(len(image_id_map) + 1)
            image_ids.append(image_id_map[orig_image_id])
        image_id = image_id_map[orig_image_id]

        dets = res.get("detections", []) or []
        for det in dets:
            full_name = _normalize_full_name(det.get("tag") or det.get("label") or "")
            parts = _split_label(full_name)
            base_class = parts["base_class"]
            subtype = parts["subtype"]
            base_classes.add(base_class)

            # Count per image_id & base_class starting from 1
            cls_counter = img_class_counters.setdefault(image_id, {})
            cls_counter[base_class] = cls_counter.get(base_class, 0) + 1
            num = str(cls_counter[base_class])

            key = f"{image_id}#{base_class}#{num}"
            record = {
                "image_id": image_id,
                "num": num,
                "label": full_name,  # full name
                "full_name": full_name,
                "tag": det.get("tag"),
                "base_class": base_class,
                "subtype": subtype,
                "bbox": det.get("bbox"),
                "diag": det.get("diag"),
                "center": det.get("center"),
            }
            inst_lookup[key] = record

            by_image_base.setdefault(image_id, {}).setdefault(base_class, []).append(key)
            by_base_image.setdefault(base_class, {}).setdefault(image_id, []).append(key)

    store = {
        "inst_lookup": inst_lookup,
        "by_image_base": by_image_base,
        "by_base_image": by_base_image,
        "image_ids": image_ids,
        "base_classes": sorted(base_classes),
    }
    return store


def save_store(store: Dict, path: str) -> None:
    """Save store to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def load_store(path: str) -> Dict:
    """Load store from JSON."""
    # Accept UTF-8 with or without BOM for compatibility with edited JSON files.
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_labels(store: Dict, image_id: str) -> List[Dict[str, str]]:
    """
    Return labels for a given image_id:
    [{base_class, subtype, full_name, num}, ...]
    """
    result = []
    for base_class, inst_keys in store.get("by_image_base", {}).get(image_id, {}).items():
        for inst_key in inst_keys:
            rec = store["inst_lookup"].get(inst_key) or {}
            result.append(
                {
                    "base_class": rec.get("base_class", base_class),
                    "subtype": rec.get("subtype", ""),
                    "full_name": rec.get("full_name") or rec.get("label", ""),
                    "num": rec.get("num", ""),
                }
            )
    return result


