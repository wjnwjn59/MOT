import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import argparse

class MVTDClassifier:
    def __init__(self, model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct", gpu_memory_utilization: float = 0.9, batch_size: int = 8):
        """
        Initialize the vLLM-based MVTD classifier for tracking challenges
        
        Args:
            model_name: Model name/path
            gpu_memory_utilization: GPU memory utilization ratio for vLLM
            batch_size: Number of frames to process in each batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Initialize vLLM engine with batch processing settings
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=32768, # Min number of tokens for multi-modal input
            max_num_seqs=batch_size,  # Enable batch processing
            dtype="auto",
            # enable_lora=True,
        )
        
        # Define the 10 tracking challenge classes
        self.classes = [
            "Occlusion",                # 0: Object bị che khuất
            "Illumination Change",      # 1: Thay đổi ánh sáng
            "Scale Variation",          # 2: Thay đổi kích thước
            "Motion Blur",              # 3: Mờ do chuyển động
            "Variance in Appearance",   # 4: Thay đổi hình dạng/màu sắc
            "Partial Visibility",       # 5: Hiển thị một phần
            "Low Resolution",           # 6: Độ phân giải thấp
            "Background Clutter",       # 7: Nền phức tạp
            "Low Contrast Object",      # 8: Đối tượng có độ tương phản thấp
            "Normal"                    # 9: Bình thường, không có thách thức đặc biệt
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

    def prepare_image_for_vllm(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert image to PIL Image for vLLM multi-modal input"""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image type")

    def create_classification_prompt(self) -> str:
        """Create classification prompt for tracking challenges (Qwen2.5-VL format)"""
        question = """You are an expert in single object tracking and computer vision. You will analyze two images:
1. Template image: Contains the target object to be tracked (cropped from first frame)
2. Current frame: Full frame where we need to track the object

Your task is to identify the main tracking challenge present in this frame compared to the template. Classify into ONE of these 10 categories:

0. Occlusion - Object is partially or fully blocked by other objects
1. Illumination Change - Significant lighting differences affect object appearance
2. Scale Variation - Object size has changed significantly from template
3. Motion Blur - Object appears blurred due to fast movement
4. Variance in Appearance - Object appearance changed (pose, orientation, deformation)
5. Partial Visibility - Object is only partially visible in the frame
6. Low Resolution - Object appears with very low resolution/pixelated
7. Background Clutter - Complex background makes object hard to distinguish
8. Low Contrast Object - Object has poor contrast with background
9. Normal - No significant tracking challenges present

Analyze the tracking difficulty and respond with only the class number (0-9) and brief explanation.
Format: ClassID: [number] - [class_name] - [explanation why this challenge is present]"""

        # Qwen2.5-VL prompt format
        prompt = (f"<|im_start|>system\nYou are a helpful assistant specialized in computer vision and object tracking.<|im_end|>\n"
                 f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                 f"<|vision_start|><|image_pad|><|vision_end|>"
                 f"{question}<|im_end|>\n"
                 f"<|im_start|>assistant\n")
        
        return prompt

    def parse_classification_result(self, response: str) -> int:
        """Parse model response to extract class ID"""
        try:
            import re
            
            # Look for ClassID pattern
            class_id_match = re.search(r'ClassID:\s*(\d+)', response, re.IGNORECASE)
            if class_id_match:
                class_id = int(class_id_match.group(1))
                if 0 <= class_id <= 9:
                    return class_id
            
            # Look for number at beginning
            number_match = re.search(r'^(\d+)', response.strip())
            if number_match:
                class_id = int(number_match.group(1))
                if 0 <= class_id <= 9:
                    return class_id
            
            return 9  # Default to Normal if unclear
            
        except Exception:
            return 9

    def load_ground_truth(self, sequence_path: str) -> Dict:
        """Load ground truth bounding boxes from sequence"""
        gt_file = os.path.join(sequence_path, 'groundtruth.txt')
        if not os.path.exists(gt_file):
            return {}
        
        gt_boxes = {}
        with open(gt_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        # Handle both comma-separated and space-separated formats
                        if ',' in line:
                            coords = list(map(float, line.split(',')))
                        else:
                            coords = list(map(float, line.split()))
                        
                        if len(coords) >= 4:
                            gt_boxes[i + 1] = coords[:4]  # Frame numbering starts from 1
                    except ValueError as e:
                        print(f"Warning: Could not parse ground truth line {i+1}: '{line}' - {e}")
                        continue
        
        return gt_boxes

    def crop_template(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop template from image using bounding box"""
        h, w = image.shape[:2]
        
        # Handle different bbox formats
        if len(bbox) == 4:
            x, y, w_box, h_box = bbox
            # Convert to integer coordinates
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + w_box))
            y2 = min(h, int(y + h_box))
        else:
            return image  # Return full image if bbox format unclear
        
        # Crop the template
        template = image[y1:y2, x1:x2]
        
        # Ensure template is not empty
        if template.size == 0:
            return image
        
        return template

    def classify_frames_batch(self, template_img: Image.Image, frame_imgs: List[Image.Image]) -> List[str]:
        """Classify multiple frames in batch for better performance"""
        # Prepare batch inputs
        batch_inputs = []
        prompt = self.create_classification_prompt()
        
        for frame_img in frame_imgs:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [template_img, frame_img]  # Template first, then current frame
                }
            }
            batch_inputs.append(inputs)
        
        # Generate responses in batch
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        # Extract responses
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        
        return responses

    def process_sequence(self, sequence_path: str) -> Dict:
        """Process a single MVTD sequence"""
        sequence_name = os.path.basename(sequence_path)
        print(f"Processing sequence: {sequence_name}")
        
        # Get all frame files
        img_files = sorted(glob.glob(os.path.join(sequence_path, '*.jpg')) + 
                          glob.glob(os.path.join(sequence_path, '*.png')))
        
        if len(img_files) == 0:
            print(f"No image files found in {sequence_path}")
            return {}
        
        # Load ground truth
        gt_boxes = self.load_ground_truth(sequence_path)
        if not gt_boxes:
            print(f"No ground truth found for {sequence_name}")
            return {}
        
        # Load first frame and create template
        first_frame = cv2.imread(img_files[0])
        if first_frame is None:
            print(f"Could not load first frame: {img_files[0]}")
            return {}
        
        # Get template from first frame
        if 1 in gt_boxes:
            template = self.crop_template(first_frame, gt_boxes[1])
        else:
            print(f"No ground truth for first frame in {sequence_name}")
            return {}
        
        # Store classification results
        results = {
            'sequence_name': sequence_name,
            'total_frames': len(img_files),
            'template_bbox': gt_boxes[1],
            'frame_classifications': {}
        }
        
        # Prepare template image once
        template_img = self.prepare_image_for_vllm(template)
        
        # Process frames in batches
        total_frames = len(img_files)
        for batch_start in range(0, total_frames, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_frames)
            batch_files = img_files[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//self.batch_size + 1}/{(total_frames + self.batch_size - 1)//self.batch_size} "
                  f"(frames {batch_start + 1}-{batch_end})")
            
            # Load frames in current batch
            batch_frames = []
            batch_frame_nums = []
            batch_filenames = []
            
            for frame_idx, img_file in enumerate(batch_files):
                frame_num = batch_start + frame_idx + 1
                current_frame = cv2.imread(img_file)
                if current_frame is not None:
                    frame_img = self.prepare_image_for_vllm(current_frame)
                    batch_frames.append(frame_img)
                    batch_frame_nums.append(frame_num)
                    batch_filenames.append(os.path.basename(img_file))
            
            if not batch_frames:
                continue
            
            # Classify batch of frames
            try:
                responses = self.classify_frames_batch(template_img, batch_frames)
                
                # Process batch results
                for i, (frame_num, filename, response) in enumerate(zip(batch_frame_nums, batch_filenames, responses)):
                    class_id = self.parse_classification_result(response)
                    
                    results['frame_classifications'][frame_num] = {
                        'frame_file': filename,
                        'class_id': class_id,
                        'class_name': self.classes[class_id],
                        'raw_response': response,
                        'ground_truth_bbox': gt_boxes.get(frame_num, None)
                    }
                
            except Exception as e:
                print(f"Error processing batch {batch_start//self.batch_size + 1}: {str(e)}")
                # Add error entries for failed batch
                for i, (frame_num, filename) in enumerate(zip(batch_frame_nums, batch_filenames)):
                    results['frame_classifications'][frame_num] = {
                        'frame_file': filename,
                        'class_id': 9,  # Default to Normal
                        'class_name': self.classes[9],
                        'error': str(e)
                    }
        
        return results

    def process_mvtd_dataset(self, dataset_path: str, output_file: Union[str, None] = None) -> Dict:
        """Process entire MVTD dataset"""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Find all sequence directories
        sequence_dirs = []
        for root, dirs, files in os.walk(dataset_path):
            # Check if this directory contains image files and ground truth
            has_images = any(f.endswith(('.jpg', '.png')) for f in files)
            has_gt = 'groundtruth.txt' in files
            
            if has_images and has_gt:
                sequence_dirs.append(root)
        
        if not sequence_dirs:
            print("No valid sequences found in dataset")
            return {}
        
        print(f"Found {len(sequence_dirs)} sequences to process")
        
        # Process all sequences
        all_results = {
            'dataset_path': dataset_path,
            'total_sequences': len(sequence_dirs),
            'processing_timestamp': str(torch.utils.data.get_worker_info()),
            'model_name': self.model_name,
            'classes': self.classes,
            'sequences': {}
        }
        
        for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
            try:
                seq_results = self.process_sequence(seq_dir)
                if seq_results:
                    seq_name = seq_results['sequence_name']
                    all_results['sequences'][seq_name] = seq_results
            except Exception as e:
                print(f"Error processing sequence {seq_dir}: {str(e)}")
        
        # Save results
        if output_file is None:
            output_file = f"mvtd_classification_results_{len(all_results['sequences'])}_sequences.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results

    def print_summary(self, results: Dict):
        """Print classification summary"""
        print("\n" + "="*60)
        print("CLASSIFICATION SUMMARY")
        print("="*60)
        
        total_frames = 0
        class_counts = {i: 0 for i in range(10)}
        
        for seq_name, seq_data in results['sequences'].items():
            frames = seq_data.get('frame_classifications', {})
            total_frames += len(frames)
            
            for frame_data in frames.values():
                class_id = frame_data.get('class_id', 9)
                class_counts[class_id] += 1
        
        print(f"Total sequences processed: {len(results['sequences'])}")
        print(f"Total frames classified: {total_frames}")
        print("\nClass distribution:")
        
        for class_id, count in class_counts.items():
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            print(f"  {class_id}. {self.classes[class_id]}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="MVTD Dataset Classification using Qwen2.5 VL")
    parser.add_argument("--dataset_path", type=str, default="/mnt/VLAI_data/MVTD/",
                       help="Path to MVTD dataset")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-VL-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--gpu_memory", type=float, default=0.9,
                       help="GPU memory utilization ratio")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing frames")
    
    args = parser.parse_args()
    
    print("Initializing MVTD Classifier...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize classifier
    classifier = MVTDClassifier(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory,
        batch_size=args.batch_size
    )
    
    # Process dataset
    try:
        results = classifier.process_mvtd_dataset(
            dataset_path=args.dataset_path,
            output_file=args.output_file
        )
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()