from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image
from tqdm import tqdm

from modules.maritime_analyzer.oracles import compute_oracle_attributes
from modules.maritime_analyzer.taxonomy import oracle_attributes, vlm_attributes, SCHEMA_VERSION

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

def shard_sequences(seq_names, num_shards):
    num_shards = max(1, int(num_shards))
    shards = [[] for _ in range(num_shards)]
    for i, s in enumerate(seq_names):
        shards[i % num_shards].append(s)
    return shards


def plan_gpu_groups(gpus, tp):
    tp = max(1, int(tp))
    return [list(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]


def build_worker_commands(num_shards, gpu_groups, dataset, out_dir, model, tp, seed):
    # One worker per shard; each must map to a GPU replica. More shards than
    # replicas would silently leave shards unprocessed while workers still
    # believe there are `num_shards` shards -> guard against silent data loss.
    if num_shards > len(gpu_groups):
        raise ValueError(
            f"num_shards ({num_shards}) exceeds available GPU replicas ({len(gpu_groups)}); "
            f"every shard needs a replica or it is never processed")
    cmds = []
    for shard_index, group in enumerate(gpu_groups[:num_shards]):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in group)
        argv = [sys.executable, "-m", "modules.maritime_analyzer.run",
                "--worker",
                "--shard-index", str(shard_index),
                "--num-shards", str(num_shards),
                "--dataset", str(dataset),
                "--out-dir", str(out_dir),
                "--model", str(model),
                "--tp", str(tp),
                "--seed", str(seed)]
        cmds.append((env, argv))
    return cmds


def build_record(seq_name, frame_id, frame_file, template_bbox, gt_bbox,
                 oracle_attrs, vlm_result, meta):
    attributes = {}
    for name in oracle_attributes():
        attributes[name] = {"prob": float(oracle_attrs[name]), "source": "oracle"}
    for name in vlm_attributes():
        attributes[name] = {"prob": float(vlm_result.get(name, 0.0)), "source": "vlm"}
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_name": seq_name,
        "frame_id": frame_id,
        "frame_file": frame_file,
        "template_bbox": list(map(float, template_bbox)),
        "ground_truth_bbox": list(map(float, gt_bbox)),
        "attributes": attributes,
        "severity": float(vlm_result.get("severity", 0.0)),
        "vlm_agreement": float(vlm_result.get("vlm_agreement", 0.0)),
        "oracle_features": oracle_attrs.get("_features", {}),
        "dataset_path": meta.get("dataset_path", ""),
    }


def _process_one_sequence_v2(seq_dir, vlm, seq_out_dir, meta):
    from PIL import Image
    frames = sorted([p for p in seq_dir.glob('*.jpg')])
    gts = read_groundtruth_txt(seq_dir / 'groundtruth.txt')
    assert len(gts) == len(frames), f"GT/frames mismatch in {seq_dir}"
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    seq_jsonl = seq_out_dir / f"{seq_dir.name}.jsonl"
    template_img, template_bbox = frames[0], gts[0]
    template_crop_path = ensure_template_crop(template_img, template_bbox, seq_out_dir)
    processed = parse_processed_frame_ids(seq_jsonl)

    for frame_id, (frame_path, bbox) in enumerate(zip(frames, gts), start=1):
        if frame_id in processed:
            continue
        frame_pil = Image.open(frame_path).convert('RGB')
        oracle = compute_oracle_attributes(
            Image.open(template_img).convert('RGB'), frame_pil,
            template_bbox, bbox, frame_pil.size)
        vlm_result = vlm.classify_soft(str(template_crop_path), str(frame_path), bbox)
        rec = build_record(seq_dir.name, frame_id, frame_path.name,
                           template_bbox, bbox, oracle, vlm_result, meta)
        with open(seq_jsonl, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--model', default='Qwen/Qwen3.5-35B-A3B')
    ap.add_argument('--gpus', default='0', help='comma list, e.g. 0,1,2,3 (orchestrator mode)')
    ap.add_argument('--tp', type=int, default=1, help='tensor-parallel size per replica')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--worker', action='store_true', help='internal: run one shard')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    sequences = sorted([p for p in dataset_path.iterdir()
                        if p.is_dir() and (p / 'groundtruth.txt').exists()])
    split_name = dataset_path.name
    out_root = Path(args.out_dir) / f"{split_name}_maritime_env_clf_annts"
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"dataset_path": str(dataset_path), "model_name": args.model, "classes": CLASSES}

    if args.worker:
        from modules.maritime_analyzer.vlm_analyzer import VLMAnalyzer, VLMConfig
        seq_names = [p.name for p in sequences]
        my_seqs = shard_sequences(seq_names, args.num_shards)[args.shard_index]
        vlm = VLMAnalyzer(VLMConfig(model_name=args.model))
        for name in my_seqs:
            seq = dataset_path / name
            try:
                _process_one_sequence_v2(seq, vlm, out_root / name, meta)
            except Exception as e:
                print(f"  !! error processing {name}: {e}")
        return

    # Orchestrator: split GPUs into replicas (TP groups), launch one worker per replica
    import subprocess
    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    groups = plan_gpu_groups(gpus, args.tp)
    num_shards = len(groups)
    cmds = build_worker_commands(num_shards, groups, args.dataset, args.out_dir,
                                 args.model, args.tp, args.seed)
    procs = [subprocess.Popen(argv, env=env) for env, argv in cmds]
    for p in procs:
        p.wait()
    print(f"All workers finished. Annotations under: {out_root}")

if __name__ == '__main__':
    main()
