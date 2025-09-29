#!/usr/bin/env python3
"""
Script to extract and save templates from the first frame of MVTD sequences.
Templates are cropped from the first frame using ground truth bounding boxes.
"""

import os
import cv2
import numpy as np
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class TemplateExtractor:
    def __init__(self, padding_ratio: float = 0.1):
        """
        Initialize template extractor
        
        Args:
            padding_ratio: Extra padding around bounding box (ratio of bbox size)
        """
        self.padding_ratio = padding_ratio

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

    def crop_template_with_padding(self, image: np.ndarray, bbox: List[float], 
                                 padding_ratio: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Crop template from image using bounding box with optional padding
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            padding_ratio: Extra padding ratio (if None, use self.padding_ratio)
            
        Returns:
            Cropped template image and crop info dict
        """
        if padding_ratio is None:
            padding_ratio = self.padding_ratio
            
        h, w = image.shape[:2]
        
        # Handle different bbox formats
        if len(bbox) == 4:
            x, y, w_box, h_box = bbox
            
            # Add padding
            padding_w = int(w_box * padding_ratio)
            padding_h = int(h_box * padding_ratio)
            
            # Calculate padded coordinates
            x1 = max(0, int(x - padding_w))
            y1 = max(0, int(y - padding_h))
            x2 = min(w, int(x + w_box + padding_w))
            y2 = min(h, int(y + h_box + padding_h))
        else:
            # Return full image if bbox format unclear
            return image, {
                'original_bbox': bbox,
                'crop_coords': [0, 0, w, h],
                'error': 'Invalid bbox format'
            }
        
        # Crop the template
        template = image[y1:y2, x1:x2]
        
        # Ensure template is not empty
        if template.size == 0:
            return image, {
                'original_bbox': bbox,
                'crop_coords': [0, 0, w, h],
                'error': 'Empty crop'
            }
        
        # Crop information for reference
        crop_info = {
            'original_bbox': bbox,
            'crop_coords': [x1, y1, x2, y2],
            'crop_size': [x2 - x1, y2 - y1],
            'padding_ratio': padding_ratio,
            'original_image_size': [w, h]
        }
        
        return template, crop_info

    def process_sequence(self, sequence_path: str, output_dir: str, 
                        save_crop_info: bool = True) -> Dict:
        """
        Process a single MVTD sequence to extract template
        
        Args:
            sequence_path: Path to sequence directory
            output_dir: Output directory for templates
            save_crop_info: Whether to save crop information as JSON
            
        Returns:
            Processing result dict
        """
        sequence_name = os.path.basename(sequence_path)
        print(f"Processing sequence: {sequence_name}")
        
        # Create output directory for this sequence
        seq_output_dir = os.path.join(output_dir, sequence_name)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # Get all frame files
        img_files = sorted(glob.glob(os.path.join(sequence_path, '*.jpg')) + 
                          glob.glob(os.path.join(sequence_path, '*.png')))
        
        if len(img_files) == 0:
            return {
                'sequence_name': sequence_name,
                'status': 'error',
                'error': 'No image files found'
            }
        
        # Load ground truth
        gt_boxes = self.load_ground_truth(sequence_path)
        if not gt_boxes or 1 not in gt_boxes:
            return {
                'sequence_name': sequence_name,
                'status': 'error',
                'error': 'No ground truth for first frame'
            }
        
        # Load first frame
        first_frame_path = img_files[0]
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            return {
                'sequence_name': sequence_name,
                'status': 'error',
                'error': f'Could not load first frame: {first_frame_path}'
            }
        
        # Extract template from first frame
        first_frame_bbox = gt_boxes[1]
        template, crop_info = self.crop_template_with_padding(first_frame, first_frame_bbox)
        
        # Save template image
        template_filename = f"{sequence_name}_template.jpg"
        template_path = os.path.join(seq_output_dir, template_filename)
        success = cv2.imwrite(template_path, template)
        
        if not success:
            return {
                'sequence_name': sequence_name,
                'status': 'error',
                'error': f'Failed to save template: {template_path}'
            }
        
        # Save crop information
        result = {
            'sequence_name': sequence_name,
            'status': 'success',
            'first_frame_path': first_frame_path,
            'template_path': template_path,
            'template_filename': template_filename,
            'crop_info': crop_info,
            'total_frames': len(img_files)
        }
        
        if save_crop_info:
            import json
            info_filename = f"{sequence_name}_template_info.json"
            info_path = os.path.join(seq_output_dir, info_filename)
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"  Template saved: {template_filename}")
        print(f"  Template size: {template.shape[1]}x{template.shape[0]}")
        
        return result

    def process_dataset(self, dataset_path: str, output_dir: str, 
                       sequences: Optional[List[str]] = None) -> Dict:
        """
        Process entire MVTD dataset or specific sequences
        
        Args:
            dataset_path: Path to MVTD dataset
            output_dir: Output directory for all templates
            sequences: List of specific sequences to process (None for all)
            
        Returns:
            Processing results dict
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find sequence directories
        if sequences:
            # Process specific sequences
            sequence_dirs = []
            for seq_name in sequences:
                seq_path = os.path.join(dataset_path, seq_name)
                if os.path.exists(seq_path):
                    sequence_dirs.append(seq_path)
                else:
                    print(f"Warning: Sequence not found: {seq_path}")
        else:
            # Find all sequences
            sequence_dirs = []
            for root, dirs, files in os.walk(dataset_path):
                # Check if this directory contains image files and ground truth
                has_images = any(f.endswith(('.jpg', '.png')) for f in files)
                has_gt = 'groundtruth.txt' in files
                
                if has_images and has_gt:
                    sequence_dirs.append(root)
        
        if not sequence_dirs:
            print("No valid sequences found")
            return {'status': 'error', 'error': 'No sequences found'}
        
        print(f"Found {len(sequence_dirs)} sequences to process")
        
        # Process all sequences
        results = {
            'dataset_path': dataset_path,
            'output_dir': output_dir,
            'total_sequences': len(sequence_dirs),
            'successful': 0,
            'failed': 0,
            'sequences': {}
        }
        
        for seq_dir in tqdm(sequence_dirs, desc="Extracting templates"):
            try:
                seq_result = self.process_sequence(seq_dir, output_dir)
                seq_name = seq_result['sequence_name']
                results['sequences'][seq_name] = seq_result
                
                if seq_result['status'] == 'success':
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                seq_name = os.path.basename(seq_dir)
                print(f"Error processing sequence {seq_name}: {str(e)}")
                results['sequences'][seq_name] = {
                    'sequence_name': seq_name,
                    'status': 'error',
                    'error': str(e)
                }
                results['failed'] += 1
        
        # Save overall results
        summary_path = os.path.join(output_dir, 'template_extraction_summary.json')
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self.print_summary(results)
        
        return results

    def print_summary(self, results: Dict):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("TEMPLATE EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"Total sequences processed: {results['total_sequences']}")
        print(f"Successful extractions: {results['successful']}")
        print(f"Failed extractions: {results['failed']}")
        
        if results['failed'] > 0:
            print("\nFailed sequences:")
            for seq_name, seq_data in results['sequences'].items():
                if seq_data['status'] == 'error':
                    error_msg = seq_data.get('error', 'Unknown error')
                    print(f"  - {seq_name}: {error_msg}")
        
        print(f"\nTemplates saved in: {results['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Extract templates from MVTD dataset first frames")
    parser.add_argument("--dataset_path", type=str, default="/mnt/VLAI_data/MVTD/test",
                       help="Path to MVTD dataset")
    parser.add_argument("--output_dir", type=str, default="./extracted_templates",
                       help="Output directory for template images")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                       help="Specific sequences to process (e.g., 1-Boat 2-ship). If not specified, process all.")
    parser.add_argument("--padding_ratio", type=float, default=0.1,
                       help="Extra padding around bounding box (ratio of bbox size)")
    parser.add_argument("--no_crop_info", action="store_true",
                       help="Don't save crop information JSON files")
    
    args = parser.parse_args()
    
    print("Initializing Template Extractor...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Padding ratio: {args.padding_ratio}")
    if args.sequences:
        print(f"Processing sequences: {args.sequences}")
    else:
        print("Processing all sequences")
    
    # Initialize extractor
    extractor = TemplateExtractor(padding_ratio=args.padding_ratio)
    
    # Process dataset
    try:
        results = extractor.process_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sequences=args.sequences
        )
        print("\nTemplate extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()