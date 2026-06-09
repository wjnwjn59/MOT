"""Audit the OLD VLM partial-visibility labels against the deterministic out-of-frame oracle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from modules.maritime_analyzer.oracles import is_out_of_frame


def compare_partial_visibility(records: List[Dict],
                               size_lookup: Callable[[str, str], tuple],
                               margin: int = 1) -> Dict:
    total = agree = disagree = 0
    vlm_partial_oracle_inside = oracle_partial_vlm_no = 0
    for r in records:
        vlm_flag = int(r["vlm_partial_flag"])
        W, H = size_lookup(r["sequence_name"], r["frame_file"])
        oracle_flag = int(is_out_of_frame(tuple(r["ground_truth_bbox"]), (W, H), margin=margin))
        total += 1
        if vlm_flag == oracle_flag:
            agree += 1
        else:
            disagree += 1
            if vlm_flag == 1 and oracle_flag == 0:
                vlm_partial_oracle_inside += 1
            elif vlm_flag == 0 and oracle_flag == 1:
                oracle_partial_vlm_no += 1
    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "disagreement_rate": (disagree / total) if total else 0.0,
        "vlm_partial_oracle_inside": vlm_partial_oracle_inside,
        "oracle_partial_vlm_no": oracle_partial_vlm_no,
    }


def _load_old_records(ann_dir: Path) -> List[Dict]:
    out = []
    for jf in ann_dir.glob("*/*.jsonl"):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip truncated/corrupt lines rather than aborting the whole audit
                pv = d.get("vlm_response", {}).get("partial_visibility", {})
                out.append({
                    "sequence_name": d.get("sequence_name", jf.parent.name),
                    "frame_file": d.get("frame_file", ""),
                    "vlm_partial_flag": int(pv.get("flag", 0)) if isinstance(pv, dict) else 0,
                    "ground_truth_bbox": d.get("ground_truth_bbox", [0, 0, 0, 0]),
                    "dataset_path": d.get("dataset_path", ""),
                })
    return out


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True, help="old-format annotation root (…_maritime_env_clf_annts)")
    ap.add_argument("--margin", type=int, default=1)
    args = ap.parse_args()
    records = _load_old_records(Path(args.ann_dir))

    ds_by_key = {(r["sequence_name"], r["frame_file"]): r["dataset_path"] for r in records}
    cache: Dict = {}
    def size_lookup(seq, frame_file):
        ds = ds_by_key.get((seq, frame_file), "")
        if not ds:
            raise ValueError(f"missing dataset_path for {seq}/{frame_file}; cannot resolve image size")
        key = (ds, seq, frame_file)
        if key not in cache:
            cache[key] = Image.open(Path(ds) / seq / frame_file).size
        return cache[key]

    stats = compare_partial_visibility(records, size_lookup, margin=args.margin)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
