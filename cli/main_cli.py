import argparse
import contextlib
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


try:
    # Unbuffer stdout for easier log watching.
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from core import config
from storage import store
from storage import memories
from agents import agent_runner
from rules import finalize
from rules import rule_translator
from rules.primitives import evaluate_rules_structured


def _load_json(path: str) -> Any:
    if not path:
        raise ValueError("JSON path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_detection_results_json(path: str) -> List[Dict]:
    data = _load_json(path)
    if isinstance(data, dict):
        results = data.get("results")
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unsupported detection JSON format: {path}")
    if not isinstance(results, list):
        raise ValueError(f"Detection JSON must contain a results list: {path}")

    normalized = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Detection result #{idx} is not an object: {path}")
        if not isinstance(item.get("detections", []), list):
            raise ValueError(f"Detection result #{idx} has invalid detections: {path}")
        normalized.append(item)
    return normalized


def _strip_suffix_digits(name: str) -> str:
    return re.sub(r"\d+$", "", name or "")


def _base_class_from_label(text: str) -> str:
    if not text:
        return ""
    base = text
    if " " in text:
        parts = text.split()
        base_token = parts[-1]
        m = re.match(r"(.+?)(\d+)$", base_token)
        base = m.group(1) if m else base_token
    elif "_" in text:
        parts = text.split("_")
        base_token = parts[-1]
        m = re.match(r"(.+?)(\d+)$", base_token)
        base = m.group(1) if m else base_token
    else:
        m = re.match(r"(.+?)(\d+)$", text)
        base = m.group(1) if m else text
    return _strip_suffix_digits(base)


def _save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def _build_llm_cfg() -> Dict:
    cfg = {"model": config.LLM_MODEL_DEFAULT, "base_url": config.LLM_BASE_URL_DEFAULT,
           "api_key": os.getenv(config.LLM_API_KEY_ENV, ""), "temperature": 0}
    required = {"model": "MARAD_LLM_MODEL", "base_url": "MARAD_LLM_BASE_URL",
                "api_key": config.LLM_API_KEY_ENV}
    missing = [env_name for field, env_name in required.items() if not cfg.get(field)]
    if missing:
        raise ValueError("Missing remote API configuration: " + ", ".join(missing))
    return cfg


def _load_or_build_store(
    results: List[Dict],
    output_dir: str,
    force_rebuild: bool = False,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    store_path = os.path.join(output_dir, "store.json")
    if os.path.exists(store_path) and not force_rebuild:
        store_data = store.load_store(store_path)
        print(f"Store loaded: {store_path}", flush=True)
        return store_data

    store_data = store.build_store(results)
    store.save_store(store_data, store_path)
    print(f"Store saved: {store_path}", flush=True)
    return store_data


def _build_detection_count_summary(
    results: List[Dict],
    mode: Optional[str] = None,
) -> Dict:
    images = []
    aggregate_label_counts: Dict[str, int] = {}
    aggregate_base_class_counts: Dict[str, int] = {}

    def _inc(bucket: Dict[str, int], key: str) -> None:
        if not key:
            return
        bucket[key] = bucket.get(key, 0) + 1

    for res in results:
        label_counts: Dict[str, int] = {}
        base_class_counts: Dict[str, int] = {}
        detections = res.get("detections", []) or []
        for det in detections:
            if not isinstance(det, dict):
                continue
            label = (det.get("label") or det.get("tag") or "").strip()
            tag = (det.get("tag") or label).strip()
            base_class = _base_class_from_label(tag or label)
            _inc(label_counts, label)
            _inc(base_class_counts, base_class)
            _inc(aggregate_label_counts, label)
            _inc(aggregate_base_class_counts, base_class)

        objects = []
        for label in sorted(label_counts.keys()):
            objects.append(
                {
                    "label": label,
                    "base_class": _base_class_from_label(label),
                    "count": label_counts[label],
                }
            )

        images.append(
            {
                "image_id": res.get("image_id"),
                "image_path": res.get("image_path"),
                "total_objects": sum(label_counts.values()),
                "objects": objects,
                "label_counts": dict(sorted(label_counts.items())),
                "base_class_counts": dict(sorted(base_class_counts.items())),
            }
        )

    return {
        "mode": mode or "",
        "observed_labels": sorted(aggregate_label_counts),
        "total_images": len(images),
        "aggregate_label_counts": dict(sorted(aggregate_label_counts.items())),
        "aggregate_base_class_counts": dict(sorted(aggregate_base_class_counts.items())),
        "images": images,
    }


def _save_detection_count_summary(
    results: List[Dict],
    output_dir: str,
    mode: Optional[str] = None,
) -> str:
    path = os.path.join(output_dir, "detection_counts.json")
    _save_json(path, _build_detection_count_summary(results, mode=mode))
    print(f"Detection count summary saved: {path}", flush=True)
    return path


def _infer_summary_payload(summary: List[Dict], abnormal_cnt: int) -> Dict:
    total = len(summary)
    return {
        "total": total,
        "abnormal": abnormal_cnt,
        "normal": total - abnormal_cnt,
        "images": summary,
    }


def _evaluate_detections(detections: List[Dict], cfg: Dict) -> Tuple[bool, List[str], Dict]:
    """
    - abnormal=True means rule failures detected.
    - issues is a list of rule failure messages.
    """
    structured = evaluate_rules_structured(detections, cfg)
    if structured.get("overall_pass", True):
        return False, [], structured

    issues = []
    for fr in structured.get("failed_rules", []):
        rid = fr.get("rule_id")
        msg = fr.get("message") or ""
        if rid:
            issues.append(f"rule[{rid}] {msg}".strip())
        else:
            issues.append(msg.strip())
    return True, [i for i in issues if i], structured


def _load_rules_cfg(rules_json: Optional[str] = None) -> Dict:
    rules_path = rules_json or config.get_rules_path()
    cfg = _load_json(rules_path)
    print(f"Rules loaded: {rules_path}", flush=True)
    return cfg


def _run_infer(detections_json: str, cfg: Dict, output_dir: Optional[str] = None) -> Dict:
    output_dir = config.configure_run("infer", output_dir)
    results = _load_detection_results_json(detections_json)
    if not results:
        raise ValueError(f"No detection results in JSON: {detections_json}")
    summary_path = os.path.join(output_dir, "infer_summary.json")
    summary = []
    abnormal_cnt = 0
    for res in results:
        abnormal, issues, _structured = _evaluate_detections(res.get("detections", []), cfg)
        abnormal_cnt += int(abnormal)
        summary.append({"image_path": res.get("image_path"), "image_id": res.get("image_id"),
                        "status": "ABNORMAL" if abnormal else "OK", "issues": issues})
        _save_json(summary_path, _infer_summary_payload(summary, abnormal_cnt))
    counts_path = _save_detection_count_summary(results, output_dir, mode="infer")
    _save_json(summary_path, _infer_summary_payload(summary, abnormal_cnt))
    print(f"Infer summary saved: {summary_path}", flush=True)
    return {"total": len(summary), "abnormal": abnormal_cnt,
            "normal": len(summary) - abnormal_cnt, "summary_path": summary_path,
            "counts_path": counts_path}


def _silence_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def run_train(args) -> None:
    llm_cfg = _build_llm_cfg()
    constraints = ""
    with _silence_stdout():
        output_dir = config.configure_run("train", args.output_dir)
        results = _load_detection_results_json(args.train_detections_json)
        if not results:
            raise ValueError(f"No train detection results in JSON: {args.train_detections_json}")
        _save_detection_count_summary(results, output_dir, mode="train")
        data = _load_or_build_store(results, output_dir, force_rebuild=True)
        memories.reset_memories()
        agent_runner.run_agent1(data, llm_cfg)
        constraints = finalize.finalize_constraints(config.RULE_MEMORY_PATH, include_fail=False)
        _save_text(os.path.join(output_dir, "constraints.txt"), constraints)
        cfg, _raw = rule_translator.translate_rule_memory(config.RULE_MEMORY_PATH, llm_cfg)
        rule_translator.save_rules_config(cfg, output_dir)
    if constraints:
        print(constraints, flush=True)


def run_infer(args) -> None:
    cfg = _load_rules_cfg(args.infer_rules_json)
    _run_infer(args.infer_detections_json, cfg, args.output_dir)


def _run_stats(detections_json: str, output_dir: Optional[str] = None) -> Dict:
    results = _load_detection_results_json(detections_json)
    if not results:
        raise ValueError(f"No detection results in JSON: {detections_json}")
    output_dir = config.configure_run("stats", output_dir)
    counts_path = _save_detection_count_summary(results, output_dir, mode="stats")
    return {"total": len(results), "counts_path": counts_path}


def _load_batch_jobs(manifest_path: str, mode: str) -> List[Dict]:
    """Read explicit jobs; relative paths are resolved beside the manifest."""
    jobs = _load_json(manifest_path)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch manifest must be a non-empty list of jobs.")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    normalized = []
    output_dirs = set()
    for index, job in enumerate(jobs):
        allowed = {"detections_json", "output_dir"}
        if mode == "infer":
            allowed.add("rules_json")
        if not isinstance(job, dict) or set(job) - allowed:
            raise ValueError(f"Batch job #{index} has unsupported fields.")
        paths = {}
        for key in allowed:
            value = job.get(key)
            if value is None and key == "rules_json":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Batch job #{index} requires {key}.")
            paths[key] = os.path.abspath(os.path.join(manifest_dir, value))
        output_key = os.path.normcase(paths["output_dir"])
        if output_key in output_dirs:
            raise ValueError("Batch jobs must have distinct output directories.")
        output_dirs.add(output_key)
        normalized.append(paths)
    return normalized


def run_batch(args) -> None:
    jobs = _load_batch_jobs(args.batch_manifest, args.mode)
    results = []
    for job in jobs:
        if args.mode == "infer":
            cfg = _load_rules_cfg(job.get("rules_json"))
            result = _run_infer(job["detections_json"], cfg, job["output_dir"])
        else:
            result = _run_stats(job["detections_json"], job["output_dir"])
        results.append({"detections_json": job["detections_json"], **result})
    output_dir = config.configure_run(args.mode, args.output_dir)
    path = os.path.join(output_dir, "batch_summary.json")
    _save_json(path, {"results": results})
    print(f"Batch summary saved: {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Counting Agent pipeline using detection JSON")
    parser.add_argument("--mode", choices=["train", "infer", "stats"], required=True,
                        help="train: rule induction; infer: rule evaluation; stats: object counts")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory; defaults to the mode directory under MARAD_RESULT_DIR.")
    parser.add_argument("--train-detections-json", default=None,
                        help="Train mode: normal reference detections JSON.")
    parser.add_argument("--infer-detections-json", default=None,
                        help="Infer mode: query detections JSON.")
    parser.add_argument("--infer-rules-json", default=None,
                        help="Infer mode: rules JSON; defaults to rules.json under MARAD_RULES_DIR.")
    parser.add_argument("--stats-detections-json", default=None,
                        help="Stats mode: detections JSON.")
    parser.add_argument("--batch-manifest", default=None,
                        help="Infer/stats mode: JSON list of jobs with detections_json, output_dir, "
                             "and optional rules_json (infer only). Paths are relative to the manifest.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode_options = {"train": ("train_detections_json",),
                    "infer": ("infer_detections_json", "infer_rules_json"),
                    "stats": ("stats_detections_json",)}
    for mode, options in mode_options.items():
        if mode != args.mode:
            for name in options:
                if getattr(args, name) is not None:
                    raise ValueError(f"--{name.replace('_', '-')} is only allowed in {mode} mode.")
    if args.batch_manifest:
        if args.mode == "train":
            raise ValueError("--batch-manifest is only allowed in infer/stats mode.")
        if any(getattr(args, name) is not None for name in mode_options[args.mode]):
            raise ValueError("Single-file options cannot be used with --batch-manifest.")
        run_batch(args)
    else:
        detection_option = f"{args.mode}_detections_json"
        if not getattr(args, detection_option):
            raise ValueError(f"--{detection_option.replace('_', '-')} is required.")
        if args.mode == "train":
            run_train(args)
        elif args.mode == "infer":
            run_infer(args)
        else:
            _run_stats(args.stats_detections_json, args.output_dir)


if __name__ == "__main__":
    main()
