import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


STATUS_OK = {"ok"}
STATUS_ABNORMAL = {"abnormal"}


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _status_to_score(status: str) -> Optional[float]:
    status_norm = (status or "").strip().lower()
    if status_norm in STATUS_OK:
        return 1.0
    if status_norm in STATUS_ABNORMAL:
        return 0.0
    return None


def _auc_rank(y_true: List[int], y_score: List[float]) -> Optional[float]:
    pairs = list(zip(y_score, y_true))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    rank = 1
    sum_pos_ranks = 0.0
    n_pos = 0
    n_neg = 0
    i = 0
    while i < n:
        j = i + 1
        score = pairs[i][0]
        while j < n and pairs[j][0] == score:
            j += 1
        count = j - i
        avg_rank = (rank + (rank + count - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                sum_pos_ranks += avg_rank
                n_pos += 1
            else:
                n_neg += 1
        rank += count
        i = j
    if n_pos == 0 or n_neg == 0:
        return None
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_summary(normal_summaries: List[str], anomaly_summaries: List[str]) -> Dict:
    """Evaluate binary rule decisions using explicitly supplied ground-truth groups.

    Normal is the positive class, matching the OK=1 / ABNORMAL=0 score convention.
    """
    y_true: List[int] = []
    y_score: List[float] = []
    unknown_status = 0
    gt_normal = gt_abnormal = pred_normal = pred_abnormal = 0
    for label, paths in ((1, normal_summaries), (0, anomaly_summaries)):
        for path in paths:
            data = _load_json(path)
            images = data.get("images")
            if not isinstance(images, list) or any(not isinstance(img, dict) for img in images):
                raise ValueError(f"Inference summary must contain an images list: {path}")
            if label == 1:
                gt_normal += len(images)
            else:
                gt_abnormal += len(images)
            for img in images:
                score = _status_to_score(img.get("status", ""))
                if score is None:
                    unknown_status += 1
                    continue
                pred_normal += int(score == 1.0)
                pred_abnormal += int(score == 0.0)
                y_true.append(label)
                y_score.append(score)
    return {"auc": _auc_rank(y_true, y_score), "n": len(y_true),
            "n_pos": sum(y_true), "n_neg": len(y_true) - sum(y_true),
            "unknown_status": unknown_status, "gt_normal": gt_normal,
            "gt_abnormal": gt_abnormal, "pred_normal": pred_normal,
            "pred_abnormal": pred_abnormal}


def main() -> None:
    parser = argparse.ArgumentParser(description="AUROC of binary rule decisions from explicit summaries")
    parser.add_argument("--normal-summary", nargs="+", required=True,
                        help="Inference summary JSON files for ground-truth normal samples.")
    parser.add_argument("--anomaly-summary", nargs="+", required=True,
                        help="Inference summary JSON files for ground-truth anomalous samples.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    args = parser.parse_args()
    result = compute_summary(args.normal_summary, args.anomaly_summary)
    out_path = args.output or os.path.join(os.getenv("MARAD_RESULT_DIR") or "result", "infer", "auroc_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": [result]}, f, indent=2, ensure_ascii=False)
    print(f"Saved AUROC summary: {out_path}")
    print(f"auc={result['auc']} n={result['n']} unknown_status={result['unknown_status']}")


if __name__ == "__main__":
    main()
