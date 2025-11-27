import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm

# Add project path
prj_path = os.path.dirname(os.path.abspath(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from simtrack_with_classification import DummyConfig, SimTrackWithClassification
from lib.models.stark.backbone import build_backbone_simtrack
from lib.models.stark.head import build_box_head
from lib.config.simtrack.config import cfg, update_config_from_file
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box

class SimpleInference:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize inference model"""
        self.device = device
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load config and build components
        yaml_file = os.path.join(os.path.dirname(__file__), 'experiments/simtrack/baseline.yaml')
        if os.path.exists(yaml_file):
            update_config_from_file(yaml_file)

        cfg = DummyConfig()

        # Build backbone and box_head
        backbone = build_backbone_simtrack(cfg)
        box_head = build_box_head(cfg)
        
        # Load model
        self.model = SimTrackWithClassification(backbone, box_head, num_classes=10)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['net']
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best accuracy: {checkpoint.get('best_accuracy', 'unknown'):.2f}%")
        else:
            self.model.load_state_dict(checkpoint, strict=True)
            
        self.model.to(device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor()
        
        # Classification mapping
        self.class_mapping = {
            0: 'motion_blur', 1: 'illu_change', 2: 'variance_appear', 
            3: 'partial_visibility', 4: 'background_clutter', 5: 'occlusion',
            6: 'scale_variation', 7: 'low_res', 8: 'low_contrast', 9: 'normal'
        }
        
        # Tracking parameters
        self.template_factor = 2.0
        self.template_size = 112
        self.search_factor = 4.5  
        self.search_size = 224
        
        print(f"Model initialized on {device}")
        
    def initialize_sequence(self, first_frame, init_bbox):
        """Initialize tracking with first frame and bounding box"""
        # Process template
        z_patch_arr, rz_factor, z_amask_arr = sample_target(
            first_frame, init_bbox, self.template_factor, output_sz=self.template_size
        )
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        # Template annotation
        bbox_sz = torch.tensor(init_bbox[2:]) * rz_factor
        template_anno = torch.tensor([
            int(self.template_size / 2 - bbox_sz[0] / 2), 
            int(self.template_size / 2 - bbox_sz[1] / 2),
            bbox_sz[0], bbox_sz[1]
        ]).to(self.device)
        
        self.template = template
        self.template_anno = template_anno / self.template_size
        self.state = init_bbox.copy()
        
    def track_frame(self, image):
        """Track object in current frame and classify environment"""
        H, W, _ = image.shape
        
        # Process search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.search_factor, output_sz=self.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # Inference
        with torch.no_grad():
            # Forward backbone
            backbone_output = self.model.forward_backbone([
                self.template, search, self.template_anno
            ])
            
            # Forward heads
            seq_dict = [backbone_output]
            head_output = self.model.forward_head(
                seq_dict, run_box_head=True, run_cls_head=True
            )
            
            # Get predictions
            pred_boxes = head_output['pred_boxes'].view(-1, 4)
            cls_logits = head_output.get('cls_logits')
            
            # Box prediction (take mean of all predictions)
            pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            
            # Classification prediction
            if cls_logits is not None:
                cls_probs = torch.softmax(cls_logits, dim=1)
                cls_pred = torch.argmax(cls_probs, dim=1)
                cls_confidence = torch.max(cls_probs, dim=1)[0]
                
                predicted_class = self.class_mapping.get(cls_pred.item(), 'unknown')
                class_confidence = cls_confidence.item()
            else:
                predicted_class = 'unknown'
                class_confidence = 0.0
                
        return self.state, predicted_class, class_confidence
    
    def map_box_back(self, pred_box, resize_factor):
        """Map predicted box back to image coordinates"""
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def load_sequence_images(sequence_dir):
    """Load all images from sequence directory"""
    img_files = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))
    images = []
    
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    return images

def read_ground_truth(gt_path):
    """Read ground truth bounding box from first line"""
    with open(gt_path, 'r') as f:
        line = f.readline().strip()
        bbox = [float(x) for x in line.split(',')]
        return bbox

def run_sequence_inference(inference_model, sequence_dir, output_dir, sequence_name):
    """Run inference on a complete sequence"""
    
    print(f"Processing sequence: {sequence_name}")
    
    # Load images
    images = load_sequence_images(sequence_dir)
    if not images:
        print(f"No images found in {sequence_dir}")
        return
        
    # Read ground truth for initialization
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
            # Track frame
            bbox, predicted_class, class_confidence = inference_model.track_frame(image)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Store results (convert to int for bbox to match original format)
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
    
    print(f"✅ Saved results to {seq_output_dir}")
    
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
    # Configuration
    checkpoint_path = "./models/SimTrackMod/checkpoints/prediction_only/simtrack_prediction_best.pth.tar"
    test_data_dir = "../data/MVTD/test"
    output_dir = "./models/SimTrackMod/output/test/tracking_results"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== Maritime Classification Inference ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Check paths
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
        
    if not os.path.exists(test_data_dir):
        print(f"❌ Test data not found: {test_data_dir}")
        return
    
    # Initialize inference model
    inference_model = SimpleInference(checkpoint_path, device)
    
    # Get all sequences
    sequences = [d for d in os.listdir(test_data_dir) 
                if os.path.isdir(os.path.join(test_data_dir, d))]
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
            print(f"❌ Error processing {sequence_name}: {e}")
            continue
    
    total_time = time.time() - total_time
    
    print("="*60)
    print(f"✅ Completed inference on {len(sequences)} sequences")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Results saved to: {output_dir}")
    print()
    print("Output structure:")
    print("sequence_name/")
    print("├── sequence_name_001.txt        # Bounding boxes")
    print("├── sequence_name_time.txt       # Inference times")  
    print("├── sequence_name_classification.txt  # Class predictions")
    print("└── sequence_name_confidence.txt # Classification confidence")

if __name__ == '__main__':
    main()