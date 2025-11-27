#!/usr/bin/env python3
"""
Script to convert model predictions to visualized images.
Visualizes tracking results by drawing bounding boxes from predictions and ground truth on images.
Selects 5 evenly distributed frames from each sequence for visualization.
"""

import os
import cv2
import numpy as np
import glob
import argparse
from pathlib import Path

def read_bounding_boxes(file_path):
    """
    Read bounding boxes from text file.
    Format: x,y,width,height per line
    Returns: list of (x, y, w, h) tuples
    """
    boxes = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    coords = [float(x) for x in line.split(',')]
                    if len(coords) == 4:
                        boxes.append(tuple(coords))
    return boxes

def draw_bounding_box(image, bbox, color, thickness=2):
    """
    Draw bounding box on image (without label).
    Args:
        image: OpenCV image
        bbox: (x, y, w, h) tuple
        color: BGR color tuple
        thickness: line thickness
    """
    x, y, w, h = [int(coord) for coord in bbox]
    
    # Draw rectangle only
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def draw_legend(image, legends):
    """
    Draw legend at bottom-left corner of image.
    Args:
        image: OpenCV image
        legends: list of (color, label) tuples
    """
    height, width = image.shape[:2]
    legend_height = 30
    legend_width = 200
    
    # Background for legend
    legend_bg = np.zeros((legend_height * len(legends) + 10, legend_width, 3), dtype=np.uint8)
    legend_bg[:] = (0, 0, 0)  # Black background with transparency effect
    
    # Add legends
    for i, (color, label) in enumerate(legends):
        y_pos = 20 + i * legend_height
        
        # Draw colored rectangle (legend marker)
        cv2.rectangle(legend_bg, (10, y_pos - 10), (30, y_pos + 5), color, -1)
        
        # Draw text
        cv2.putText(legend_bg, label, (40, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Position legend at bottom-left
    legend_h, legend_w = legend_bg.shape[:2]
    start_y = height - legend_h - 10
    start_x = 10
    
    # Blend legend with image (semi-transparent effect)
    alpha = 0.8
    roi = image[start_y:start_y + legend_h, start_x:start_x + legend_w]
    blended = cv2.addWeighted(roi, 1 - alpha, legend_bg, alpha, 0)
    image[start_y:start_y + legend_h, start_x:start_x + legend_w] = blended

def get_frame_indices(total_frames, num_samples=5):
    """
    Get evenly distributed frame indices.
    Args:
        total_frames: total number of frames
        num_samples: number of samples to select
    Returns: list of frame indices
    """
    if total_frames <= num_samples:
        return list(range(total_frames))
    
    step = total_frames // (num_samples + 1)
    indices = [step * (i + 1) for i in range(num_samples)]
    return indices

def process_sequence(dataset_path, predictions_path, sequence_name, output_dir):
    """
    Process a single sequence to create visualization images.
    Args:
        dataset_path: path to dataset directory
        predictions_path: path to predictions directory
        sequence_name: name of the sequence
        output_dir: output directory for visualized images
    """
    print(f"Processing sequence: {sequence_name}")
    
    # Paths
    seq_dataset_dir = os.path.join(dataset_path, sequence_name)
    seq_pred_dir = os.path.join(predictions_path, sequence_name)
    seq_output_dir = os.path.join(output_dir, sequence_name)
    
    # Create output directory
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # Get image files
    image_files = sorted(glob.glob(os.path.join(seq_dataset_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {seq_dataset_dir}")
        return
    
    total_frames = len(image_files)
    print(f"Found {total_frames} frames")
    
    # Read ground truth
    gt_file = os.path.join(seq_dataset_dir, "groundtruth.txt")
    gt_boxes = read_bounding_boxes(gt_file)
    
    # Find prediction file
    pred_files = glob.glob(os.path.join(seq_pred_dir, f"{sequence_name}_*.txt"))
    pred_boxes = []
    if pred_files:
        # Take the first matching prediction file
        pred_file = pred_files[0]
        pred_boxes = read_bounding_boxes(pred_file)
        print(f"Using prediction file: {os.path.basename(pred_file)}")
    else:
        print(f"No prediction file found for {sequence_name}")
    
    # Get frame indices to visualize
    frame_indices = get_frame_indices(total_frames, num_samples=5)
    print(f"Selected frame indices: {frame_indices}")
    
    # Process selected frames
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(image_files):
            continue
            
        # Read image
        image_path = image_files[frame_idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Prepare legend
        legends = []
        
        # Draw ground truth box (green)
        if frame_idx < len(gt_boxes):
            draw_bounding_box(image, gt_boxes[frame_idx], (0, 255, 0), thickness=3)
            legends.append(((0, 255, 0), "Ground Truth"))
        
        # Draw prediction box (red)
        if frame_idx < len(pred_boxes):
            draw_bounding_box(image, pred_boxes[frame_idx], (0, 0, 255), thickness=2)
            legends.append(((0, 0, 255), "Prediction"))
        
        # Draw legend at bottom-left corner
        if legends:
            draw_legend(image, legends)
        
        # Add frame info
        frame_text = f"Frame: {frame_idx + 1}/{total_frames}"
        cv2.putText(image, frame_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save image
        output_filename = f"{sequence_name}_frame_{frame_idx + 1:06d}_segment_{i + 1}.jpg"
        output_path = os.path.join(seq_output_dir, output_filename)
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Convert model predictions to visualized images")
    parser.add_argument("--dataset_path", type=str, default="/mnt/VLAI_data/MVTD/test",
                       help="Path to dataset directory")
    parser.add_argument("--predictions_path", type=str, 
                       default=None,
                       help="Path to predictions directory")
    parser.add_argument("--output_dir", type=str, default="./visualization_output",
                       help="Output directory for visualized images")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                       help="Specific sequences to process (e.g., 1-Boat 2-ship). If not specified, process all.")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequences to process
    if args.sequences:
        sequences = args.sequences
    else:
        # Get all sequences from predictions directory
        sequences = [d for d in os.listdir(args.predictions_path) 
                    if os.path.isdir(os.path.join(args.predictions_path, d))]
        sequences.sort()
    
    print(f"Processing {len(sequences)} sequences: {sequences}")
    
    # Process each sequence
    for sequence in sequences:
        try:
            process_sequence(args.dataset_path, args.predictions_path, 
                           sequence, args.output_dir)
            print(f"Completed: {sequence}\n")
        except Exception as e:
            print(f"Error processing {sequence}: {str(e)}\n")
    
    print("All sequences processed!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()