import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import argparse

# Add project path
prj_path = os.path.dirname(os.path.abspath(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from simtrack_with_classification import DummyConfig, SimTrackWithClassification
from lib.models.stark.backbone import build_backbone_simtrack
from lib.models.stark.head import build_box_head
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box


class SimpleInference:
    def __init__(self, checkpoint_path, device='cuda', use_cls=False):
        """
        Initialize inference model.

        Args:
            checkpoint_path (str): Path to checkpoint saved by finetune script.
            device (str): 'cuda' or 'cpu'.
            use_cls (bool): If True, run classification head (only valid if the
                            model was trained with cls head enabled).
                            If False, run pure tracking (bbox only).
        """
        self.device = device
        self.use_cls = use_cls

        print(f"Loading model from: {checkpoint_path}")
        print(f"Classification head enabled at inference: {self.use_cls}")

        # Build config consistent with training (DummyConfig from simtrack_with_classification)
        cfg = DummyConfig()

        # Build backbone and box_head
        backbone = build_backbone_simtrack(cfg)
        box_head = build_box_head(cfg)

        # Build extended model
        self.model = SimTrackWithClassification(backbone, box_head, num_classes=10)

        # ---- Robust checkpoint loading ----
        full_ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Try common formats:
        if isinstance(full_ckpt, dict):
            if 'net' in full_ckpt:
                state_dict = full_ckpt['net']
                print(f"Found 'net' in checkpoint. Epoch: {full_ckpt.get('epoch', 'unknown')}")
            elif 'model_state_dict' in full_ckpt:
                state_dict = full_ckpt['model_state_dict']
                print(f"Found 'model_state_dict' in checkpoint. Epoch: {full_ckpt.get('epoch', 'unknown')}")
            else:
                # Assume it's directly a state_dict
                state_dict = full_ckpt
                print("Checkpoint looks like a raw state_dict.")
        else:
            # Very old/bare format: directly a state_dict
            state_dict = full_ckpt
            print("Checkpoint is a raw state_dict (non-dict wrapper).")

        self.model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully.")

        self.model.to(device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = Preprocessor()

        # Classification mapping (must match training)
        self.class_mapping = {
            0: 'motion_blur',
            1: 'illu_change',
            2: 'variance_appear',
            3: 'partial_visibility',
            4: 'background_clutter',
            5: 'occlusion',
            6: 'scale_variation',
            7: 'low_res',
            8: 'low_contrast',
            9: 'normal',
        }

        # Tracking parameters
        self.template_factor = 2.0
        self.template_size = 112
        self.search_factor = 4.5
        self.search_size = 224

        print(f"Model initialized on {device}")

    def initialize_sequence(self, first_frame, init_bbox):
        """Initialize tracking with first frame and bounding box (x, y, w, h)."""
        # Process template patch
        z_patch_arr, rz_factor, z_amask_arr = sample_target(
            first_frame, init_bbox, self.template_factor, output_sz=self.template_size
        )
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)  # (1, 3, H, W)

        # Approximate bbox in template coordinates (xywh in template patch)
        bbox_sz = torch.tensor(init_bbox[2:], dtype=torch.float32) * rz_factor
        template_anno_xywh = torch.tensor([
            self.template_size / 2.0 - bbox_sz[0] / 2.0,
            self.template_size / 2.0 - bbox_sz[1] / 2.0,
            bbox_sz[0],
            bbox_sz[1],
        ], dtype=torch.float32)

        # Normalize to [0,1] (still xywh). This matches how you used it in training
        # in your current pipeline, even though comments sometimes say cxcywh.
        template_anno_norm = template_anno_xywh / float(self.template_size)

        # Store state
        self.template = template.to(self.device)
        self.template_anno = template_anno_norm.to(self.device)
        self.state = init_bbox.copy()

    def track_frame(self, image):
        """Track object in current frame and (optionally) classify environment."""
        H, W, _ = image.shape

        # Process search region around current state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.search_factor, output_sz=self.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)  # (1, 3, H, W)

        predicted_class = 'unknown'
        class_confidence = 0.0

        # Inference
        with torch.no_grad():
            # Forward backbone
            backbone_output = self.model.forward_backbone(
                [self.template, search.to(self.device), self.template_anno]
            )

            # Forward heads
            seq_dict = [backbone_output]
            head_output = self.model.forward_head(
                seq_dict,
                run_box_head=True,
                run_cls_head=self.use_cls,  # IMPORTANT: respect training mode
            )

            # ---- Box prediction ----
            pred_boxes = head_output['pred_boxes'].view(-1, 4)  # (B*N, 4), usually (1,4)
            # Take mean in case of multiple queries
            pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()

            # Map box back to original image coordinates
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

            # ---- Classification prediction (optional) ----
            if self.use_cls and 'cls_logits' in head_output:
                cls_logits = head_output['cls_logits']
                cls_probs = torch.softmax(cls_logits, dim=1)
                cls_pred = torch.argmax(cls_probs, dim=1)
                cls_conf = torch.max(cls_probs, dim=1)[0]

                predicted_class = self.class_mapping.get(cls_pred.item(), 'unknown')
                class_confidence = cls_conf.item()
            else:
                predicted_class = 'unknown'
                class_confidence = 0.0

        return self.state, predicted_class, class_confidence

    def map_box_back(self, pred_box, resize_factor):
        """
        Map predicted box in search patch coordinates back to image coordinates.
        pred_box is [cx, cy, w, h] in search patch space.
        """
        # Current state center in original image
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]

        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def load_sequence_images(sequence_dir):
    """Load all images from sequence directory."""
    img_files = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))
    images = []

    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return images


def read_ground_truth(gt_path):
    """Read ground truth bounding box (x,y,w,h) from first line."""
    with open(gt_path, 'r') as f:
        line = f.readline().strip()
        bbox = [float(x) for x in line.split(',')]
        return bbox


def run_sequence_inference(inference_model, sequence_dir, output_dir, sequence_name):
    """Run inference on a complete sequence."""
    print(f"Processing sequence: {sequence_name}")

    images = load_sequence_images(sequence_dir)
    if not images:
        print(f"No images found in {sequence_dir}")
        return

    gt_path = os.path.join(sequence_dir, "groundtruth.txt")
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return

    init_bbox = read_ground_truth(gt_path)

    # Initialize sequence
    inference_model.initialize_sequence(images[0], init_bbox)

    # Prepare output directory
    seq_output_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(seq_output_dir, exist_ok=True)

    # Output files
    bbox_file = os.path.join(seq_output_dir, f"{sequence_name}_001.txt")
    time_file = os.path.join(seq_output_dir, f"{sequence_name}_time.txt")
    class_file = os.path.join(seq_output_dir, f"{sequence_name}_classification.txt")
    confidence_file = os.path.join(seq_output_dir, f"{sequence_name}_confidence.txt")

    bbox_results = []
    time_results = []
    class_results = []
    confidence_results = []

    # Track through sequence
    for i, image in enumerate(tqdm(images, desc=f"Tracking {sequence_name}")):
        start_time = time.time()

        if i == 0:
            # First frame - use initialization
            bbox = init_bbox
            predicted_class = 'initialization'
            class_confidence = 1.0
        else:
            bbox, predicted_class, class_confidence = inference_model.track_frame(image)

        end_time = time.time()
        inference_time = end_time - start_time

        # Store results (bbox as ints to match original format)
        bbox_results.append(f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}")
        time_results.append(f"{inference_time:.6f}")
        class_results.append(predicted_class)
        confidence_results.append(f"{class_confidence:.6f}")

    # Save results
    with open(bbox_file, 'w') as f:
        f.write('\n'.join(bbox_results) + '\n')

    with open(time_file, 'w') as f:
        f.write('\n'.join(time_results) + '\n')

    with open(class_file, 'w') as f:
        f.write('\n'.join(class_results) + '\n')

    with open(confidence_file, 'w') as f:
        f.write('\n'.join(confidence_results) + '\n')

    print(f"[SUCCESS] Saved results to {seq_output_dir}")

    # Print classification summary
    class_counts = {}
    for cls in class_results:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("Classification summary:")
    for cls, count in sorted(class_counts.items()):
        percentage = count / len(class_results) * 100
        print(f"  {cls}: {count} frames ({percentage:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Maritime Tracking + Classification Inference')
    parser.add_argument(
        '--checkpoint', type=str,
        default='./checkpoints/maritime_classification/simtrack_prediction_best.pth.tar',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_data', type=str,
        default='../data/MVTD/test',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--output', type=str,
        default='./output/test/tracking_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (cuda/cpu/auto). Default: auto'
    )
    parser.add_argument(
        '--use_cls', action='store_true',
        help='Run classification head at inference. '
             'Enable this ONLY if the checkpoint was trained with cls head.'
    )

    args = parser.parse_args()

    # Device resolution
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    checkpoint_path = args.checkpoint
    test_data_dir = args.test_data
    output_dir = args.output

    print("Maritime Tracking Inference")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Use classification head: {args.use_cls}")
    print()

    # Path checks
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    if not os.path.exists(test_data_dir):
        print(f"[ERROR] Test data not found: {test_data_dir}")
        return

    # Initialize inference model
    inference_model = SimpleInference(checkpoint_path, device=device, use_cls=args.use_cls)

    # Get all sequences
    sequences = [
        d for d in os.listdir(test_data_dir)
        if os.path.isdir(os.path.join(test_data_dir, d))
    ]
    sequences.sort()

    print(f"Found {len(sequences)} sequences: {sequences}")
    print()

    # Process each sequence
    total_time = time.time()

    for sequence_name in sequences:
        sequence_dir = os.path.join(test_data_dir, sequence_name)
        try:
            run_sequence_inference(inference_model, sequence_dir, output_dir, sequence_name)
        except Exception as e:
            print(f"[ERROR] Error processing {sequence_name}: {e}")
            continue

    total_time = time.time() - total_time

    print("=" * 60)
    print(f"[SUCCESS] Completed inference on {len(sequences)} sequences")
    print(f"[TIME] Total time: {total_time:.2f} seconds")
    print(f"[OUTPUT] Results saved to: {output_dir}")
    print()
    print("Output structure:")
    print("sequence_name/")
    print("├── sequence_name_001.txt            # Bounding boxes")
    print("├── sequence_name_time.txt           # Inference times")
    print("├── sequence_name_classification.txt # Class predictions")
    print("└── sequence_name_confidence.txt     # Classification confidence")


if __name__ == '__main__':
    main()
