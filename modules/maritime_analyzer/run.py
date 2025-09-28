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
    """Read existing per-frame lines from .jsonl and return the set of processed frame IDs (1-based)."""
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
                # accept both "frame_id" at top-level or within "frame"
                if "frame_id" in obj:
                    done.add(int(obj["frame_id"]))
                elif "frame" in obj and "frame_id" in obj["frame"]:
                    done.add(int(obj["frame"]["frame_id"]))
            except Exception:
                # ignore malformed lines
                pass
    return done

def ensure_template_crop(template_img_path: Path, template_bbox, out_dir: Path) -> Path:
    """Create (or reuse) template crop under out_dir/template_crop.jpg and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / "template_crop.jpg"
    if crop_path.exists():
        return crop_path
    imgT = Image.open(template_img_path).convert('RGB')
    x, y, w, h = [int(round(v)) for v in template_bbox]
    cropT = imgT.crop((x, y, x + w, y + h))
    cropT.save(crop_path)
    return crop_path

def process_sequence_streaming(
    seq_dir: Path,
    vlm: VLMAnalyzer,
    seq_out_dir: Path,
    meta: Dict,
) -> None:
    """
    Stream processing: for each frame, compute labels and append one JSON line
    to <seq_out_dir>/<seq_name>.jsonl immediately.
    """
    frames = sorted([p for p in seq_dir.glob('*.jpg')])
    gt_file = seq_dir / 'groundtruth.txt'
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing groundtruth.txt in {seq_dir}")
    gts = read_groundtruth_txt(gt_file)
    assert len(gts) == len(frames), f"GT/frames mismatch in {seq_dir}"

    seq_out_dir.mkdir(parents=True, exist_ok=True)
    seq_jsonl = seq_out_dir / f"{seq_dir.name}.jsonl"

    # Template info
    template_img = frames[0]
    template_bbox = gts[0]
    template_crop_path = ensure_template_crop(template_img, template_bbox, seq_out_dir)

    # Determine already-processed frame IDs (1-based)
    processed_ids = parse_processed_frame_ids(seq_jsonl)

    # tqdm bar for frames
    iterable = list(enumerate(zip(frames, gts), start=1))
    pbar = tqdm(iterable, desc=f"{seq_dir.name}", unit="frame")
    for frame_id, (frame_path, bbox) in pbar:
        if frame_id in processed_ids:
            pbar.set_postfix_str(f"skip {frame_id}")
            continue

        # Deterministic labels
        det = analyze_pair(template_img, frame_path, template_bbox, bbox)
        # VLM labels
        vlm_flags = vlm.classify(str(template_crop_path), str(frame_path), bbox)

        # Build a single-line JSON record for this frame
        record = {
            # global meta (so each line is self-contained)
            "dataset_path": meta["dataset_path"],
            "processing_time": datetime.utcnow().isoformat(),
            "model_name": meta["model_name"],
            "classes": meta["classes"],
            # sequence context
            "sequence_name": seq_dir.name,
            "total_frames": len(frames),
            "template_bbox": list(map(float, template_bbox)),
            # frame result
            "frame_id": frame_id,  # 1-based
            "frame_file": frame_path.name,
            "cv_response": {
                "scale_variation": det.scale_variation,
                "low_res": det.low_res,
                "low_contrast": det.low_contrast,
            },
            "vlm_response": {
                "motion_blur": vlm_flags.get("motion_blur", 0),
                "illu_change": vlm_flags.get("illu_change", 0),
                "variance_appear": vlm_flags.get("variance_appear", 0),
                "partial_visibility": vlm_flags.get("partial_visibility", 0),
                "background_clutter": vlm_flags.get("background_clutter", 0),
                "occlusion": vlm_flags.get("occlusion", 0),
            },
            "ground_truth_bbox": list(map(float, bbox)),
        }

        # Append immediately
        with open(seq_jsonl, 'a') as f:
            f.write(json.dumps(record))
            f.write('\n')

        # Update progress bar
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

    # Output root: <out-dir>/<split>_<timestamp>/
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    split_name = dataset_path.name
    out_root = Path(args.out_dir) / f"{split_name}_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Sequence-level resume file (marks fully processed sequences)
    resume_path = Path(args.resume_file) if args.resume_file else (Path(args.out_dir) / f"{split_name}_processed.txt")
    processed_sequences: set[str] = set()
    if resume_path.exists():
        with open(resume_path, 'r') as f:
            processed_sequences = {line.strip() for line in f if line.strip()}
    else:
        resume_path.parent.mkdir(parents=True, exist_ok=True)

    vlm = VLMAnalyzer(VLMConfig(model_name=args.model))

    # Shared metadata per line
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
        seq_jsonl = seq_dir_out / f"{seq.name}.jsonl"

        # Process this sequence in streaming mode (per-frame appends)
        print(f"[{i}/{total}] Processing {seq.name} ...")
        try:
            process_sequence_streaming(seq, vlm, seq_dir_out, meta)
        except Exception as e:
            print(f"  !! error processing {seq.name}: {e}")
            # don't mark as complete; you can resume later and it will skip frames already in the .jsonl
            continue

        # If we got here without exceptions, mark the sequence complete
        with open(resume_path, 'a') as rf:
            rf.write(seq.name + '\n')
        print(f"  → sequence complete: {seq.name}")

    print(f"All annotations (this run) saved under: {out_root}")
    print(f"Resume file: {resume_path} (fully processed sequences listed)")

if __name__ == '__main__':
    main()
