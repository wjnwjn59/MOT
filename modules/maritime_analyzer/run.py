from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image
from tqdm import tqdm

from modules.maritime_analyzer.deterministic_utils import analyze_pair
from modules.maritime_analyzer.vlm_analyzer import VLMAnalyzer, VLMConfig

CLASSES = [
    "Occlusion",
    "Illumination Change",
    "Scale Variation",
    "Motion Blur",
    "Variance in Appearance",
    "Partial Visibility",
    "Low Resolution",
    "Background Clutter",
    "Low Contrast Object",
    "Normal",
]

def read_groundtruth_txt(fp: Path) -> List[Tuple[float, float, float, float]]:
    bboxes = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            x, y, w, h = map(float, parts)
            bboxes.append((x, y, w, h))
    return bboxes

def parse_processed_frame_ids(seq_jsonl: Path) -> Set[int]:
    done = set()
    if not seq_jsonl.exists():
        return done
    with open(seq_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "frame_id" in obj:
                    done.add(int(obj["frame_id"]))
                elif "frame" in obj and "frame_id" in obj["frame"]:
                    done.add(int(obj["frame"]["frame_id"]))
            except Exception:
                pass
    return done

def ensure_template_crop(template_img_path: Path, template_bbox, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / "template_crop.jpg"
    if crop_path.exists():
        return crop_path
    imgT = Image.open(template_img_path).convert('RGB')
    W, H = imgT.size
    x, y, w, h = [max(0, int(round(v))) for v in template_bbox]
    x = min(x, W-1)
    y = min(y, H-1)
    w = min(w, W-x)
    h = min(h, H-y)
    if w <= 0 or h <= 0:
        # Fallback to full image if bbox is invalid
        cropT = imgT
    else:
        cropT = imgT.crop((x, y, x + w, y + h))
    cropT.save(crop_path)
    return crop_path

def process_sequence_streaming(
    seq_dir: Path,
    vlm: VLMAnalyzer,
    seq_out_dir: Path,
    meta: Dict,
) -> None:
    frames = sorted([p for p in seq_dir.glob('*.jpg')])
    gt_file = seq_dir / 'groundtruth.txt'
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing groundtruth.txt in {seq_dir}")
    gts = read_groundtruth_txt(gt_file)
    assert len(gts) == len(frames), f"GT/frames mismatch in {seq_dir}"

    seq_out_dir.mkdir(parents=True, exist_ok=True)
    seq_jsonl = seq_out_dir / f"{seq_dir.name}.jsonl"

    template_img = frames[0]
    template_bbox = gts[0]
    template_crop_path = ensure_template_crop(template_img, template_bbox, seq_out_dir)

    processed_ids = parse_processed_frame_ids(seq_jsonl)

    iterable = list(enumerate(zip(frames, gts), start=1))
    pbar = tqdm(iterable, desc=f"{seq_dir.name}", unit="frame")
    for frame_id, (frame_path, bbox) in pbar:
        if frame_id in processed_ids:
            pbar.set_postfix_str(f"skip {frame_id}")
            continue

        det = analyze_pair(template_img, frame_path, template_bbox, bbox)
        vlm_flags = vlm.classify(str(template_crop_path), str(frame_path), bbox)

        # Build per-class combined (flag + confidence)
        def pack(name: str):
            return {
                "flag": int(vlm_flags.get(name, 0)),
                "conf": float(vlm_flags.get("vlm_scores", {}).get(name, 0.0)),
            }

        vlm_response = {
            "motion_blur": pack("motion_blur"),
            "illu_change": pack("illu_change"),
            "variance_appear": pack("variance_appear"),
            "partial_visibility": pack("partial_visibility"),
            "background_clutter": pack("background_clutter"),
            "occlusion": pack("occlusion"),
            "raw_confidence": vlm_flags.get("raw_confidence", 0.0),
        }

        record = {
            "dataset_path": meta["dataset_path"],
            "processing_time": datetime.utcnow().isoformat(),
            "model_name": meta["model_name"],
            "classes": meta["classes"],
            "sequence_name": seq_dir.name,
            "total_frames": len(frames),
            "template_bbox": list(map(float, template_bbox)),
            "frame_id": frame_id,
            "frame_file": frame_path.name,
            "cv_response": {
                "scale_variation": det.scale_variation,
                "low_res": det.low_res,
                "low_contrast": det.low_contrast,
            },
            "vlm_response": vlm_response,
            "ground_truth_bbox": list(map(float, bbox)),
        }

        with open(seq_jsonl, 'a') as f:
            f.write(json.dumps(record))
            f.write('\n')

        pbar.set_postfix_str(f"done {frame_id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='Path to MVTD split')
    ap.add_argument('--out-dir', default='data', help='Directory to store per-sequence .jsonl files')
    ap.add_argument('--model', default='unsloth/Qwen2-VL-7B-Instruct')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--resume-file', default=None,
                    help='Optional path to a text file that tracks fully processed sequence names. '
                         'Default: <out-dir>/<split>_processed.txt')
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    sequences = [p for p in dataset_path.iterdir() if p.is_dir() and (p / 'groundtruth.txt').exists()]
    sequences.sort()

    split_name = dataset_path.name
    out_root = Path(args.out_dir) / f"{split_name}_maritime_env_clf_annts"
    out_root.mkdir(parents=True, exist_ok=True)

    resume_path = Path(args.resume_file) if args.resume_file else (Path(args.out_dir) / f"{split_name}_processed.txt")
    processed_sequences: set[str] = set()
    if resume_path.exists():
        with open(resume_path, 'r') as f:
            processed_sequences = {line.strip() for line in f if line.strip()}
    else:
        resume_path.parent.mkdir(parents=True, exist_ok=True)

    vlm = VLMAnalyzer(VLMConfig(model_name=args.model))

    meta = {
        "dataset_path": str(dataset_path),
        "model_name": args.model,
        "classes": CLASSES,
    }

    total = len(sequences)
    for i, seq in enumerate(sequences, start=1):
        print(f"[{i}/{total}] Checking {seq.name} ...")
        if seq.name in processed_sequences:
            print("  → skip sequence (marked complete in resume file)")
            continue

        seq_dir_out = out_root / seq.name
        seq_dir_out.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{total}] Processing {seq.name} ...")
        try:
            process_sequence_streaming(seq, vlm, seq_dir_out, meta)
        except Exception as e:
            print(f"  !! error processing {seq.name}: {e}")
            continue

        with open(resume_path, 'a') as rf:
            rf.write(seq.name + '\n')
        print(f"  → sequence complete: {seq.name}")

    print(f"All annotations saved under: {out_root}")
    print(f"Resume file: {resume_path}")

if __name__ == '__main__':
    main()
